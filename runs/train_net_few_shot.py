#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

"""Train a video classification model."""
import numpy as np
import pprint
import torch
import torch.nn.functional as F
import math
import os
import oss2 as oss
import torch.nn as nn

import models.utils.losses as losses
import models.utils.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
import utils.bucket as bu
from utils.meters import TrainMeter, ValMeter
from utils.few_shot_training import (
    get_few_shot_progress,
    validate_few_shot_training_config,
)
from ipdb import set_trace
from models.base.builder import build_model
from datasets.base.builder import build_loader, shuffle_dataset

from datasets.utils.mixup import Mixup

logger = logging.get_logger(__name__)


def _validate_required_semantic_loss(model_dict, cfg):
    """在反向传播前验证启用的语义分支确实处于活动状态。"""
    semantic_weight = float(getattr(cfg.TRAIN, "SEMANTIC_LOSS_WEIGHT", 0.0))
    if not math.isfinite(semantic_weight) or semantic_weight < 0:
        raise RuntimeError(
            "SEMANTIC_LOSS_WEIGHT 必须为有限的非负数，"
            f"当前为 {semantic_weight!r}"
        )
    if semantic_weight == 0:
        return None

    if "semantic_alignment_loss" not in model_dict:
        raise RuntimeError(
            "SEMANTIC_LOSS_WEIGHT > 0，但模型没有返回 semantic_alignment_loss"
        )
    semantic_loss = model_dict["semantic_alignment_loss"]
    if not isinstance(semantic_loss, torch.Tensor):
        raise RuntimeError(
            "semantic_alignment_loss 必须是 torch.Tensor，实际为 "
            f"{type(semantic_loss).__name__}"
        )
    if semantic_loss.numel() != 1:
        raise RuntimeError(
            "semantic_alignment_loss 必须是单元素标量，实际 shape="
            f"{tuple(semantic_loss.shape)}"
        )
    if not torch.isfinite(semantic_loss).all().item():
        raise RuntimeError("semantic_alignment_loss 包含 NaN 或 Inf")

    status = model_dict.get("semantic_alignment_status")
    allow_fallback = bool(
        getattr(cfg.TRAIN, "ALLOW_SEMANTIC_FALLBACK", False)
    )
    if status == "fallback":
        if not allow_fallback:
            raise RuntimeError(
                "模型返回 semantic_alignment_status=fallback，"
                "但 ALLOW_SEMANTIC_FALLBACK=false"
            )
        if semantic_loss.detach().item() != 0.0:
            raise RuntimeError("fallback 状态的 semantic_alignment_loss 必须严格为 0")
        return semantic_loss

    if status != "active":
        raise RuntimeError(
            "SEMANTIC_LOSS_WEIGHT > 0 时 semantic_alignment_status 必须为 active，"
            f"实际为 {status!r}"
        )
    if not semantic_loss.requires_grad or semantic_loss.grad_fn is None:
        raise RuntimeError(
            "active 语义损失没有真实计算图，无法参与反向传播"
        )
    return semantic_loss


def _save_and_evaluate_few_shot(
    model,
    model_ema,
    optimizer,
    scaler,
    progress,
    cfg,
    val_meter,
    val_loader,
    writer,
    model_bucket,
):
    """Save only after a completed task and run validation on the same state."""
    if progress.tasks_done % int(cfg.TRAIN.VAL_FRE_ITER) != 0:
        return

    checkpoint_epoch = (
        progress.tasks_done // int(cfg.TRAIN.VAL_FRE_ITER)
        + int(cfg.TRAIN.NUM_FOLDS)
        - 2
    )
    train_state = cu.build_few_shot_train_state(
        cfg, progress.tasks_done, scaler
    )
    cu.save_checkpoint(
        cfg.OUTPUT_DIR,
        model,
        model_ema,
        optimizer,
        checkpoint_epoch,
        cfg,
        model_bucket,
        train_state=train_state,
    )
    logger.info(
        "Few-shot checkpoint saved after task %d/%d (pseudo epoch %.3f/%d).",
        progress.tasks_done,
        cfg.TRAIN.NUM_TRAIN_TASKS,
        float(progress.tasks_done) / cfg.SOLVER.STEPS_ITER,
        cfg.SOLVER.MAX_EPOCH,
    )

    if val_meter is None or val_loader is None:
        return

    val_meter.set_model_ema_enabled(False)
    eval_epoch(
        val_loader,
        model,
        val_meter,
        progress.pseudo_epoch,
        cfg,
        writer,
        global_step=progress.tasks_done,
    )
    if model_ema is not None:
        val_meter.set_model_ema_enabled(True)
        ema_model = model_ema.module if hasattr(model_ema, "module") else model_ema
        eval_epoch(
            val_loader,
            ema_model,
            val_meter,
            progress.pseudo_epoch,
            cfg,
            writer,
            global_step=progress.tasks_done,
        )
    model.train()


def train_epoch(
    train_loader,
    model,
    model_ema,
    optimizer,
    train_meter,
    completed_tasks,
    scaler,
    mixup_fn,
    cfg,
    writer=None,
    val_meter=None,
    val_loader=None,
    model_bucket=None,
):
    """Train the remaining few-shot tasks on one global task timeline."""
    model.train()
    use_amp = getattr(cfg.TRAIN, "USE_AMP", False)
    norm_train = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm3d, nn.LayerNorm)) and module.training:
            norm_train = True
    logger.info(f"Norm training: {norm_train}")

    steps_per_epoch = int(cfg.SOLVER.STEPS_ITER)
    last_tasks_done = int(completed_tasks)
    optimizer.zero_grad()
    train_meter.iter_tic()

    for cur_iter, task_dict in enumerate(train_loader):
        progress = get_few_shot_progress(
            completed_tasks, cur_iter, steps_per_epoch
        )
        if progress.global_task_index >= int(cfg.TRAIN.NUM_TRAIN_TASKS):
            break

        cur_epoch = progress.pseudo_epoch
        if misc.get_num_gpus(cfg):
            for k in task_dict.keys():
                task_dict[k] = task_dict[k][0].cuda(non_blocking=True)
            

        if mixup_fn is not None:
            inputs, labels["supervised_mixup"] = mixup_fn(inputs, labels["supervised"])


        # Update LR once from the zero-based global task index.
        lr = optim.get_epoch_lr(progress.lr_time, cfg)
        optim.set_lr(optimizer, lr)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

        else:
            with torch.cuda.amp.autocast(enabled=use_amp):
                model_dict = model(task_dict)

        target_logits = model_dict['logits']

        if hasattr(cfg.TRAIN,"USE_CLASSIFICATION") and cfg.TRAIN.USE_CLASSIFICATION:
            if hasattr(cfg.TRAIN,"USE_CLASSIFICATION_ONLY") and cfg.TRAIN.USE_CLASSIFICATION_ONLY:
                loss = cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long()) /cfg.TRAIN.BATCH_SIZE
            elif hasattr(cfg.TRAIN,"USE_LOCAL") and cfg.TRAIN.USE_LOCAL:
                if hasattr(cfg.TRAIN,"TEMPORAL_LOSS_WEIGHT") and cfg.TRAIN.TEMPORAL_LOSS_WEIGHT:
                    loss =  (cfg.TRAIN.TEMPORAL_LOSS_WEIGHT*model_dict["loss_temporal_regular"] + F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1).long())) /cfg.TRAIN.BATCH_SIZE
                else:
                    loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).unsqueeze(1).repeat(1,cfg.DATA.NUM_INPUT_FRAMES).reshape(-1).long())) /cfg.TRAIN.BATCH_SIZE
       
            else:
                # set_trace()
                if hasattr(cfg.TRAIN,"USE_CONTRASTIVE") and cfg.TRAIN.USE_CONTRASTIVE:
                    if hasattr(cfg.TRAIN,"USE_MOTION") and cfg.TRAIN.USE_MOTION:
                        if hasattr(cfg.TRAIN,"MOTION_COFF") and cfg.TRAIN.MOTION_COFF:
                            loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE  + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.MOTION_COFF*(F.cross_entropy(model_dict["logits_motion"], task_dict["target_labels"].long()))
                        else:
                            if hasattr(cfg.TRAIN,"USE_RECONS") and cfg.TRAIN.USE_RECONS:
                                loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE  + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.RECONS_COFF*model_dict["loss_recons"]
                            else:
                                loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE  + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s_motion"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE
                    else:
                        if hasattr(cfg.TRAIN,"USE_RECONS") and cfg.TRAIN.USE_RECONS:
                            loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.RECONS_COFF*model_dict["loss_recons"]
                        else:
                            loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_s2q"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE + cfg.TRAIN.USE_CONTRASTIVE_COFF * F.cross_entropy(model_dict["logits_q2s"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE
                else:
                    loss =  (F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) + cfg.TRAIN.USE_CLASSIFICATION_VALUE * F.cross_entropy(model_dict["class_logits"], torch.cat([task_dict["real_support_labels"], task_dict["real_target_labels"]], 0).long())) /cfg.TRAIN.BATCH_SIZE
        else:
            
            loss =  F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE
            semantic_loss = _validate_required_semantic_loss(model_dict, cfg)
            if semantic_loss is not None:
                loss = loss + cfg.TRAIN.SEMANTIC_LOSS_WEIGHT * semantic_loss / cfg.TRAIN.BATCH_SIZE
       
        # A NaN task is consumed on the global timeline, but no gradients survive.
        if torch.isnan(loss).any():
            optimizer.zero_grad()
            last_tasks_done = progress.tasks_done
            logger.warning(
                "NaN loss at few-shot task %d; gradients were cleared.",
                progress.tasks_done,
            )
            train_meter.iter_toc()
            if (
                progress.tasks_done % steps_per_epoch == 0
                and train_meter.num_samples > 0
            ):
                train_meter.log_epoch_stats(progress.pseudo_epoch)
                train_meter.reset()
            _save_and_evaluate_few_shot(
                model,
                model_ema,
                optimizer,
                scaler,
                progress,
                cfg,
                val_meter,
                val_loader,
                writer,
                model_bucket,
            )
            train_meter.iter_tic()
            continue
        scaler.scale(loss).backward(retain_graph=False)

        # optimize
        if progress.tasks_done % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0:
            if hasattr(cfg.TRAIN,"CLIP_GRAD_NORM") and cfg.TRAIN.CLIP_GRAD_NORM:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.TRAIN.CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        # self.scheduler.step()

        if hasattr(cfg, "MULTI_MODAL") and\
            cfg.PRETRAIN.PROTOTYPE.ENABLE and\
            cur_epoch < cfg.PRETRAIN.PROTOTYPE.FREEZE_EPOCHS:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        # Update the parameters.
        # optimizer.step()
        if model_ema is not None:
            model_ema.update(model)

        if cfg.DETECTION.ENABLE or cfg.PRETRAIN.ENABLE:
            if misc.get_num_gpus(cfg) > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                None, None, loss, lr, inputs["video"].shape[0] if isinstance(inputs, dict) else inputs.shape[0]
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss": loss, "Train/lr": lr},
                    global_step=progress.tasks_done,
                )
            if cfg.PRETRAIN.ENABLE:
                train_meter.update_custom_stats(loss_in_parts)

        else:
            top1_err, top5_err = None, None
            if isinstance(task_dict['target_labels'], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                if misc.get_num_gpus(cfg) > 1:
                    loss = du.all_reduce([loss])[0].item()
                    for k, v in loss_in_parts.items():
                        loss_in_parts[k] = du.all_reduce([v])[0].item()
                else:
                    loss = loss.item()
                    for k, v in loss_in_parts.items():
                        loss_in_parts[k] = v.item()
                train_meter.update_custom_stats(loss_in_parts)
                train_meter.update_custom_stats(top1_err_all)
                train_meter.update_custom_stats(top5_err_all)
            else:
                # Compute the errors.
                preds = target_logits
                num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if misc.get_num_gpus(cfg) > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                train_loader.batch_size
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=progress.tasks_done,
                )

        last_tasks_done = progress.tasks_done
        train_meter.log_iter_stats(
            progress.pseudo_epoch, progress.task_in_epoch
        )
        if progress.tasks_done % steps_per_epoch == 0:
            train_meter.log_epoch_stats(progress.pseudo_epoch)
            train_meter.reset()

        _save_and_evaluate_few_shot(
            model,
            model_ema,
            optimizer,
            scaler,
            progress,
            cfg,
            val_meter,
            val_loader,
            writer,
            model_bucket,
        )
        train_meter.iter_tic()

    if (
        last_tasks_done > int(completed_tasks)
        and last_tasks_done % steps_per_epoch != 0
        and train_meter.num_samples > 0
    ):
        train_meter.log_epoch_stats((last_tasks_done - 1) // steps_per_epoch)
        train_meter.reset()
    return last_tasks_done


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None, global_step=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    eval_global_step = cur_epoch if global_step is None else global_step
    val_meter.iter_tic()

    for cur_iter, task_dict in enumerate(val_loader):
        if cur_iter >= cfg.TRAIN.NUM_TEST_TASKS:
            break
        if misc.get_num_gpus(cfg):

            for k in task_dict.keys():
                task_dict[k] = task_dict[k][0].cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if misc.get_num_gpus(cfg):
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if misc.get_num_gpus(cfg) > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        elif cfg.PRETRAIN.ENABLE and (cfg.PRETRAIN.GENERATOR == 'PCMGenerator'):
            preds, logits = model(inputs)
            if "move_x" in preds.keys():
                preds["move_joint"] = preds["move_x"]
            elif "move_y" in preds.keys():
                preds["move_joint"] = preds["move_y"]
            num_topks_correct = metrics.topks_correct(preds["move_joint"], labels["self-supervised"]["move_joint"].reshape(preds["move_joint"].shape[0]), (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds["move_joint"].shape[0]) * 100.0 for x in num_topks_correct
            ]
            if misc.get_num_gpus(cfg) > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            val_meter.iter_toc()
            val_meter.update_stats(
                top1_err,
                top5_err,
                preds["move_joint"].shape[0]
                * max(
                    misc.get_num_gpus(cfg), 1
                ),
            )
            val_meter.update_predictions(preds, labels)
        else:
            # preds, logits = model(inputs)
            model_dict = model(task_dict)

            # loss, loss_in_parts, weight = losses.calculate_loss(cfg, preds, logits, labels, cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size)
            target_logits = model_dict['logits']
            loss =  F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long()) /cfg.TRAIN.BATCH_SIZE


            top1_err, top5_err = None, None
            if isinstance(task_dict['target_labels'], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 5))
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                val_meter.update_custom_stats(top1_err_all)
                val_meter.update_custom_stats(top5_err_all)
            else:
                # Compute the errors.
                labels = task_dict['target_labels']
                preds = target_logits
                num_topks_correct = metrics.topks_correct(preds, task_dict['target_labels'], (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if misc.get_num_gpus(cfg) > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )
            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                val_loader.batch_size
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                    global_step=eval_global_step,
                )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=eval_global_step
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if misc.get_num_gpus(cfg):
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=eval_global_step
            )

    val_meter.reset()

def train_few_shot(cfg):
    """Train few-shot tasks with SOLVER as the single progress/optimizer source."""
    validate_few_shot_training_config(cfg)

    # Set up environment.
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    model, model_ema = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    if cfg.OSS.ENABLE:
        model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        model_bucket = bu.initialize_bucket(
            cfg.OSS.KEY,
            cfg.OSS.SECRET,
            cfg.OSS.ENDPOINT,
            model_bucket_name,
        )
    else:
        model_bucket = None

    logger.info(
        "Few-shot optimizer source: SOLVER "
        "(method=%s, base_lr=%s, weight_decay=%s, max_epoch=%s).",
        cfg.SOLVER.OPTIM_METHOD,
        cfg.SOLVER.BASE_LR,
        cfg.SOLVER.WEIGHT_DECAY,
        cfg.SOLVER.MAX_EPOCH,
    )
    optimizer = optim.construct_optimizer(model, cfg, optimizer_cfg=cfg.SOLVER)

    use_amp = getattr(cfg.TRAIN, "USE_AMP", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Restore model/optimizer first, then recover the authoritative task counter.
    start_epoch, train_state = cu.load_train_checkpoint(
        cfg,
        model,
        model_ema,
        optimizer,
        model_bucket,
        return_train_state=True,
    )
    completed_tasks = cu.restore_few_shot_train_state(
        cfg, start_epoch, train_state, scaler
    )

    train_loader = build_loader(cfg, "train")
    val_loader = (
        build_loader(cfg, "test") if cfg.TRAIN.EVAL_PERIOD != 0 else None
    )

    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(
            cfg.SOLVER.STEPS_ITER,
            cfg,
            schedule_cfg=cfg.SOLVER,
        )
        val_meter = (
            ValMeter(len(val_loader), cfg, schedule_cfg=cfg.SOLVER)
            if val_loader is not None
            else None
        )

    if cfg.AUGMENTATION.MIXUP.ENABLE or cfg.AUGMENTATION.CUTMIX.ENABLE:
        logger.info("Enabling mixup/cutmix.")
        mixup_fn = Mixup(cfg)
        cfg.TRAIN.LOSS_FUNC = "soft_target"
    else:
        logger.info("Mixup/cutmix disabled.")
        mixup_fn = None

    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(misc.get_num_gpus(cfg)):
        pass
    else:
        writer = None

    logger.info(
        "Start few-shot task: %d/%d (pseudo epoch %.3f/%d).",
        completed_tasks + 1 if completed_tasks < cfg.TRAIN.NUM_TRAIN_TASKS else completed_tasks,
        cfg.TRAIN.NUM_TRAIN_TASKS,
        float(completed_tasks) / cfg.SOLVER.STEPS_ITER,
        cfg.SOLVER.MAX_EPOCH,
    )

    resume_pseudo_epoch = completed_tasks // cfg.SOLVER.STEPS_ITER
    shuffle_dataset(train_loader, resume_pseudo_epoch)

    if completed_tasks < cfg.TRAIN.NUM_TRAIN_TASKS:
        completed_tasks = train_epoch(
            train_loader,
            model,
            model_ema,
            optimizer,
            train_meter,
            completed_tasks,
            scaler,
            mixup_fn,
            cfg,
            writer,
            val_meter,
            val_loader,
            model_bucket,
        )
    else:
        logger.info("Few-shot training already completed; no task was executed.")

    logger.info(
        "Few-shot training stopped at completed_tasks=%d/%d.",
        completed_tasks,
        cfg.TRAIN.NUM_TRAIN_TASKS,
    )

    if writer is not None:
        writer.close()
    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TRAIN.LOG_FILE)
        bu.put_to_bucket(
            model_bucket,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2],
        )
