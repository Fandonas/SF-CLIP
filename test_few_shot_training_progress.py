#!/usr/bin/env python3
"""CPU tests for the task-based few-shot training timeline."""

import math
import os
import tempfile
import unittest
from types import SimpleNamespace

from models.utils import lr_policy
from utils.few_shot_training import (
    get_few_shot_progress,
    infer_legacy_completed_tasks,
    validate_few_shot_training_config,
)

try:
    import torch
except ImportError:
    torch = None


class TestConfig(SimpleNamespace):
    def dump(self):
        return "{}"


def make_cfg():
    cfg = TestConfig()
    cfg.TRAIN = SimpleNamespace(
        NUM_TRAIN_TASKS=7000,
        BATCH_SIZE_PER_TASK=2,
        VAL_FRE_ITER=100,
        NUM_FOLDS=1,
        ONLY_LINEAR=False,
        LR_REDUCE=False,
        FINE_TUNE=False,
    )
    cfg.SOLVER = SimpleNamespace(
        BASE_LR=1e-5,
        LR_POLICY="steps_with_relative_lrs",
        STEPS_ITER=250,
        STEPS=[0, 17, 25],
        LRS=[1.0, 0.1, 0.01],
        MAX_EPOCH=28,
        WARMUP_EPOCHS=1,
        WARMUP_START_LR=5e-6,
        OPTIM_METHOD="adam",
        WEIGHT_DECAY=5e-4,
    )
    cfg.OPTIMIZER = SimpleNamespace(
        BASE_LR=2e-3,
        OPTIM_METHOD="adam",
        WEIGHT_DECAY=1e-3,
    )
    cfg.BN = SimpleNamespace(WB_LOCK=True, WEIGHT_DECAY=0.0)
    cfg.NUM_GPUS = 1
    cfg.NUM_SHARDS = 1
    cfg.PAI = False
    return cfg


class FewShotTimelineTests(unittest.TestCase):
    def test_lr_key_tasks(self):
        cfg = make_cfg()
        expected = {
            1: 5e-6,
            250: 9.98e-6,
            251: 1e-5,
            4251: 1e-6,
            6251: 1e-7,
            7000: 1e-7,
        }
        for task_number, expected_lr in expected.items():
            progress = get_few_shot_progress(
                0, task_number - 1, cfg.SOLVER.STEPS_ITER
            )
            actual_lr = lr_policy.get_lr_at_epoch(cfg, progress.lr_time)
            self.assertTrue(
                math.isclose(actual_lr, expected_lr, rel_tol=0.0, abs_tol=1e-12),
                "task {}: {} != {}".format(task_number, actual_lr, expected_lr),
            )

    def test_task_251_uses_one_lr_epoch_not_two(self):
        cfg = make_cfg()
        progress = get_few_shot_progress(0, 250, cfg.SOLVER.STEPS_ITER)
        self.assertEqual(progress.lr_time, 1.0)
        self.assertEqual(lr_policy.get_lr_at_epoch(cfg, progress.lr_time), 1e-5)

    def test_resume_runs_exact_remaining_task_count(self):
        cfg = make_cfg()
        first = get_few_shot_progress(100, 0, cfg.SOLVER.STEPS_ITER)
        last = get_few_shot_progress(100, 6899, cfg.SOLVER.STEPS_ITER)
        stop = get_few_shot_progress(100, 6900, cfg.SOLVER.STEPS_ITER)
        self.assertEqual(first.tasks_done, 101)
        self.assertEqual(last.tasks_done, 7000)
        self.assertEqual(stop.global_task_index, cfg.TRAIN.NUM_TRAIN_TASKS)

    def test_current_config_is_consistent(self):
        validate_few_shot_training_config(make_cfg())

    def test_invalid_epoch_count_is_rejected(self):
        cfg = make_cfg()
        cfg.SOLVER.MAX_EPOCH = 10
        with self.assertRaisesRegex(ValueError, "MAX_EPOCH"):
            validate_few_shot_training_config(cfg)

    def test_legacy_checkpoint_falls_back_to_clean_accumulation_boundary(self):
        cfg = make_cfg()
        self.assertEqual(infer_legacy_completed_tasks(1, cfg), 98)
        self.assertEqual(infer_legacy_completed_tasks(2, cfg), 198)


@unittest.skipIf(torch is None, "PyTorch is not installed in this Python environment")
class FewShotTorchIntegrationTests(unittest.TestCase):
    def test_optimizer_source_is_explicit_only_for_few_shot(self):
        from models.utils.optimizer import construct_optimizer

        cfg = make_cfg()
        regular_model = torch.nn.Linear(2, 2)
        regular_optimizer = construct_optimizer(regular_model, cfg)
        regular_decay = [
            group["weight_decay"]
            for group in regular_optimizer.param_groups
            if len(group["params"]) > 0
        ]
        self.assertEqual(regular_decay, [cfg.OPTIMIZER.WEIGHT_DECAY])

        few_shot_model = torch.nn.Linear(2, 2)
        few_shot_optimizer = construct_optimizer(
            few_shot_model, cfg, optimizer_cfg=cfg.SOLVER
        )
        few_shot_decay = [
            group["weight_decay"]
            for group in few_shot_optimizer.param_groups
            if len(group["params"]) > 0
        ]
        self.assertEqual(few_shot_decay, [cfg.SOLVER.WEIGHT_DECAY])

    def test_new_checkpoint_round_trip_preserves_completed_tasks(self):
        if not hasattr(torch.cuda, "amp") or not hasattr(torch.cuda.amp, "GradScaler"):
            self.skipTest("This PyTorch build has no GradScaler")

        import utils.checkpoint as checkpoint

        cfg = make_cfg()
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        train_state = checkpoint.build_few_shot_train_state(cfg, 100, scaler)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = checkpoint.save_checkpoint(
                tmp_dir,
                model,
                None,
                optimizer,
                0,
                cfg,
                train_state=train_state,
            )
            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.Adam(
                restored_model.parameters(), lr=cfg.SOLVER.BASE_LR
            )
            epoch, restored_state = checkpoint.load_checkpoint(
                cfg,
                path,
                restored_model,
                None,
                data_parallel=False,
                optimizer=restored_optimizer,
                return_train_state=True,
            )
            restored_scaler = torch.cuda.amp.GradScaler(enabled=False)
            completed_tasks = checkpoint.restore_few_shot_train_state(
                cfg, epoch + 1, restored_state, restored_scaler
            )

        self.assertEqual(epoch, 0)
        self.assertEqual(completed_tasks, 100)
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()