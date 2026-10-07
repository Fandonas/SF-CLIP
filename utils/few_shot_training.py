#!/usr/bin/env python3
"""Few-shot task progress and schedule validation helpers."""

import math
from collections import namedtuple


FewShotProgress = namedtuple(
    "FewShotProgress",
    [
        "global_task_index",
        "tasks_done",
        "pseudo_epoch",
        "task_in_epoch",
        "lr_time",
    ],
)


def get_few_shot_progress(completed_tasks, local_iter, steps_per_epoch):
    """Convert resumed/local task counters into one consistent global timeline."""
    completed_tasks = int(completed_tasks)
    local_iter = int(local_iter)
    steps_per_epoch = int(steps_per_epoch)
    if completed_tasks < 0 or local_iter < 0:
        raise ValueError("Few-shot task counters must be non-negative.")
    if steps_per_epoch <= 0:
        raise ValueError("SOLVER.STEPS_ITER must be positive.")

    global_task_index = completed_tasks + local_iter
    return FewShotProgress(
        global_task_index=global_task_index,
        tasks_done=global_task_index + 1,
        pseudo_epoch=global_task_index // steps_per_epoch,
        task_in_epoch=global_task_index % steps_per_epoch,
        lr_time=float(global_task_index) / steps_per_epoch,
    )


def validate_few_shot_training_config(cfg):
    """Fail fast when task, accumulation, and LR schedule units disagree."""
    num_tasks = int(cfg.TRAIN.NUM_TRAIN_TASKS)
    steps_per_epoch = int(cfg.SOLVER.STEPS_ITER)
    max_epoch = int(cfg.SOLVER.MAX_EPOCH)
    accumulation = int(cfg.TRAIN.BATCH_SIZE_PER_TASK)
    val_frequency = int(cfg.TRAIN.VAL_FRE_ITER)
    steps = list(cfg.SOLVER.STEPS)
    relative_lrs = list(cfg.SOLVER.LRS)

    if num_tasks <= 0:
        raise ValueError("TRAIN.NUM_TRAIN_TASKS must be positive.")
    if steps_per_epoch <= 0:
        raise ValueError("SOLVER.STEPS_ITER must be positive.")
    expected_max_epoch = int(math.ceil(float(num_tasks) / steps_per_epoch))
    if max_epoch != expected_max_epoch:
        raise ValueError(
            "SOLVER.MAX_EPOCH must equal ceil(TRAIN.NUM_TRAIN_TASKS / "
            "SOLVER.STEPS_ITER): expected {}, got {}.".format(
                expected_max_epoch, max_epoch
            )
        )
    if not steps or len(steps) != len(relative_lrs):
        raise ValueError("SOLVER.STEPS and SOLVER.LRS must be non-empty and equal length.")
    if steps[0] != 0 or any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("SOLVER.STEPS must start at 0 and be strictly increasing.")
    if steps[-1] >= max_epoch:
        raise ValueError("The last SOLVER.STEPS value must be smaller than MAX_EPOCH.")
    if accumulation <= 0:
        raise ValueError("TRAIN.BATCH_SIZE_PER_TASK must be positive.")
    if val_frequency <= 0 or val_frequency % accumulation != 0:
        raise ValueError(
            "TRAIN.VAL_FRE_ITER must be positive and divisible by "
            "TRAIN.BATCH_SIZE_PER_TASK."
        )
    if num_tasks % accumulation != 0:
        raise ValueError(
            "TRAIN.NUM_TRAIN_TASKS must be divisible by TRAIN.BATCH_SIZE_PER_TASK."
        )


def build_few_shot_schedule_signature(cfg):
    """Return the optimizer/LR fields that must not drift across a new resume."""
    return {
        "base_lr": float(cfg.SOLVER.BASE_LR),
        "lr_policy": str(cfg.SOLVER.LR_POLICY),
        "steps_per_epoch": int(cfg.SOLVER.STEPS_ITER),
        "steps": [int(value) for value in cfg.SOLVER.STEPS],
        "relative_lrs": [float(value) for value in cfg.SOLVER.LRS],
        "max_epoch": int(cfg.SOLVER.MAX_EPOCH),
        "warmup_epochs": float(cfg.SOLVER.WARMUP_EPOCHS),
        "warmup_start_lr": float(cfg.SOLVER.WARMUP_START_LR),
        "optim_method": str(cfg.SOLVER.OPTIM_METHOD),
        "weight_decay": float(cfg.SOLVER.WEIGHT_DECAY),
    }


def infer_legacy_completed_tasks(start_epoch, cfg):
    """Map the legacy pre-task checkpoint index to a clean accumulation boundary."""
    start_epoch = int(start_epoch)
    if start_epoch <= 0:
        return 0

    validation_boundary = (
        start_epoch - int(cfg.TRAIN.NUM_FOLDS) + 1
    ) * int(cfg.TRAIN.VAL_FRE_ITER)
    if validation_boundary <= 0:
        return 0

    accumulation = int(cfg.TRAIN.BATCH_SIZE_PER_TASK)
    return ((validation_boundary - 1) // accumulation) * accumulation