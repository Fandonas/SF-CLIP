#!/usr/bin/env python3
"""CPU tests for strict semantic-alignment failure handling."""

import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


IMPORT_ERROR = None
torch = None
F = None
few_shot_module = None
semantic_module = None
train_module = None

try:
    import torch
    import torch.nn.functional as F

    from models.base import few_shot as few_shot_module
    from models.base import semantic_alignment_few_shot as semantic_module
except Exception as exc:  # pragma: no cover - depends on the project runtime.
    IMPORT_ERROR = exc

if semantic_module is not None:
    try:
        from runs import train_net_few_shot as train_module
    except Exception:
        train_module = None


def _runtime_skip_reason():
    if IMPORT_ERROR is None:
        return "project PyTorch runtime is unavailable"
    return "project PyTorch runtime is unavailable: {}".format(IMPORT_ERROR)


class SemanticConfigTests(unittest.TestCase):
    def test_k100_disables_semantic_fallback(self):
        config_path = (
            Path(__file__).resolve().parent
            / "configs"
            / "projects"
            / "SF-CLIP"
            / "kinetics100"
            / "k100_vit_5way1shot_newclass.yaml"
        )
        content = config_path.read_text(encoding="utf-8")
        self.assertRegex(
            content,
            re.compile(r"(?m)^\s{2}ALLOW_SEMANTIC_FALLBACK:\s*false\s*$"),
        )


@unittest.skipUnless(semantic_module is not None, _runtime_skip_reason())
class SemanticAlignmentFailFastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.classes = ["class_{}".format(index) for index in range(5)]

    def make_model(self, training=True, allow_fallback=False):
        model_class = semantic_module.CNN_SEMANTIC_ALIGNMENT_FEW_SHOT
        model = model_class.__new__(model_class)
        torch.nn.Module.__init__(model)
        model.args = SimpleNamespace(
            TRAIN=SimpleNamespace(
                WAY=5,
                WAT_TEST=5,
                SEMANTIC_COMBINE=True,
                TEXT_COFF=0.5,
            )
        )
        model.semantic_loss_weight = 1.0
        model.semantic_temperature = 0.1
        model.semantic_transpose = False
        model.allow_semantic_fallback = allow_fallback
        model.mid_dim = 3
        model.class_real_train = list(self.classes)
        model.class_real_test = list(self.classes)
        model._full_clip_model = None
        model._semantic_stages = None
        model._stage_text_features = {
            class_name: torch.randn(2, model.mid_dim)
            for class_name in self.classes
        }
        model._text_features_initialized = True
        model._semantic_initialization_error = None
        model._semantic_fallback_warnings = set()
        model.train(training)
        return model

    @staticmethod
    def fake_cos_sim(left, right):
        return F.normalize(left, dim=-1).matmul(
            F.normalize(right, dim=-1).transpose(0, 1)
        )

    @staticmethod
    def fake_otam(dists, lbda=0.5):
        del lbda
        return dists.mean(dim=-1).mean(dim=-1)

    def make_episode_inputs(self, target_count=5):
        return {
            # 与真实数据集一致：图像按视频帧展平，标签仍按视频计数。
            "support_set": torch.zeros(5 * 6, 1),
            "target_set": torch.zeros(target_count * 6, 1),
            "support_labels": torch.arange(5),
            "real_support_labels": torch.arange(5),
            "real_target_labels": torch.arange(target_count),
        }

    def test_text_model_load_failure_is_cached_and_raised(self):
        model = self.make_model()
        model._text_features_initialized = False
        model._stage_text_features = None
        model.args.VIDEO = SimpleNamespace(
            HEAD=SimpleNamespace(BACKBONE_NAME="ViT-B/16")
        )

        with mock.patch.object(
            few_shot_module, "load", side_effect=RuntimeError("load failed")
        ) as mocked_load:
            with self.assertRaisesRegex(
                semantic_module.SemanticAlignmentError, "初始化失败"
            ):
                model._ensure_text_features_initialized()
            self.assertFalse(model._text_features_initialized)
            self.assertIsNone(model._stage_text_features)
            self.assertIsNotNone(model._semantic_initialization_error)

            with self.assertRaisesRegex(
                semantic_module.SemanticAlignmentError, "不再重复加载"
            ):
                model._ensure_text_features_initialized()
            self.assertEqual(mocked_load.call_count, 1)

    def test_any_stage_encoding_failure_aborts_the_class(self):
        model = self.make_model()
        model._semantic_stages = {"complex_class": ["first", "second"]}
        model._full_clip_model = mock.Mock()
        model._full_clip_model.encode_text.side_effect = [
            torch.ones(1, model.mid_dim),
            RuntimeError("encode failed"),
        ]

        with mock.patch.object(
            few_shot_module,
            "tokenize",
            return_value=torch.zeros(1, 1, dtype=torch.long),
        ):
            with self.assertRaisesRegex(
                semantic_module.SemanticAlignmentError,
                "complex_class.*second",
            ):
                model._create_stage_text_features()

    def test_episode_requires_exact_target_labels_and_valid_ids(self):
        model = self.make_model()
        valid_inputs = self.make_episode_inputs()
        all_names, episode_names = model._build_semantic_episode_class_names(
            valid_inputs, 5, 5
        )
        self.assertEqual(len(all_names), 10)
        self.assertEqual(episode_names, self.classes)

        missing_target_labels = dict(valid_inputs)
        missing_target_labels.pop("real_target_labels")
        with self.assertRaisesRegex(
            semantic_module.SemanticAlignmentError, "real_target_labels"
        ):
            model._build_semantic_episode_class_names(
                missing_target_labels, 5, 5
            )

        short_target_labels = dict(valid_inputs)
        short_target_labels["real_target_labels"] = torch.arange(4)
        with self.assertRaisesRegex(
            semantic_module.SemanticAlignmentError, "数量.*不一致"
        ):
            model._build_semantic_episode_class_names(short_target_labels, 5, 5)

        invalid_target_id = dict(valid_inputs)
        invalid_target_id["real_target_labels"] = torch.tensor([0, 1, 2, 3, 99])
        with self.assertRaisesRegex(
            semantic_module.SemanticAlignmentError, "ID 99 越界"
        ):
            model._build_semantic_episode_class_names(invalid_target_id, 5, 5)

    def test_missing_episode_class_never_shrinks_five_way_loss(self):
        model = self.make_model()
        del model._stage_text_features[self.classes[-1]]
        video_features = torch.randn(10, 4, 3, requires_grad=True)
        class_names = self.classes * 2

        with self.assertRaisesRegex(
            semantic_module.SemanticAlignmentError, "缺少语义文本特征"
        ):
            model._compute_semantic_alignment_loss_multi_class(
                video_features, class_names
            )

    def test_nonfinite_similarity_and_otam_are_rejected(self):
        model = self.make_model()
        video_features = torch.randn(10, 4, 3, requires_grad=True)
        class_names = self.classes * 2

        def nan_similarity(left, right):
            return torch.full(
                (left.shape[0], right.shape[0]),
                float("nan"),
                device=left.device,
            )

        with mock.patch.object(semantic_module, "cos_sim", nan_similarity):
            with self.assertRaisesRegex(
                semantic_module.SemanticAlignmentError, "NaN 或 Inf"
            ):
                model._compute_semantic_alignment_loss_multi_class(
                    video_features, class_names
                )

        def inf_otam(dists, lbda=0.5):
            del lbda
            return torch.full(
                (dists.shape[0], 1), float("inf"), device=dists.device
            )

        with mock.patch.object(semantic_module, "cos_sim", self.fake_cos_sim), \
                mock.patch.object(semantic_module, "OTAM_cum_dist_v2", inf_otam):
            with self.assertRaisesRegex(
                semantic_module.SemanticAlignmentError, "NaN 或 Inf"
            ):
                model._compute_semantic_alignment_loss_multi_class(
                    video_features, class_names
                )

    def test_valid_five_way_loss_is_finite_and_backpropagates(self):
        model = self.make_model()
        video_features = torch.randn(10, 4, 3, requires_grad=True)
        class_names = self.classes * 2

        with mock.patch.object(semantic_module, "cos_sim", self.fake_cos_sim), \
                mock.patch.object(
                    semantic_module, "OTAM_cum_dist_v2", self.fake_otam
                ):
            loss = model._compute_semantic_alignment_loss_multi_class(
                video_features, class_names
            )

        self.assertEqual(loss.numel(), 1)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(loss.grad_fn)
        loss.backward()
        self.assertIsNotNone(video_features.grad)
        self.assertTrue(torch.isfinite(video_features.grad).all().item())
        self.assertGreater(video_features.grad.abs().sum().item(), 0.0)

    def test_valid_loss_matches_the_previous_formula(self):
        model = self.make_model()
        video_features = torch.randn(10, 4, 3, requires_grad=True)
        class_names = self.classes * 2

        with mock.patch.object(semantic_module, "cos_sim", self.fake_cos_sim), \
                mock.patch.object(
                    semantic_module, "OTAM_cum_dist_v2", self.fake_otam
                ):
            actual = model._compute_semantic_alignment_loss_multi_class(
                video_features, class_names
            )

            frames_flat = video_features.reshape(-1, video_features.shape[-1])
            cumulative = []
            for class_name in self.classes:
                stage_features = model._stage_text_features[class_name]
                similarity = self.fake_cos_sim(
                    frames_flat, stage_features
                ) / model.semantic_temperature
                distances = 1 - torch.sigmoid(
                    similarity.reshape(10, 4, stage_features.shape[0])
                )
                distance_4d = distances.transpose(-1, -2).unsqueeze(1)
                cumulative.append(self.fake_otam(distance_4d).squeeze(1))
            legacy_logits = -torch.stack(cumulative, dim=1)
            targets = torch.tensor([0, 1, 2, 3, 4] * 2)
            expected = F.cross_entropy(legacy_logits, targets)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0.0))

    def test_eval_missing_class_fails_instead_of_returning_visual_logits(self):
        model = self.make_model(training=False)
        del model._stage_text_features[self.classes[-1]]
        inputs = self.make_episode_inputs()
        target_features = torch.randn(5, 4, 3)
        visual_logits = torch.randn(5, 5)

        with self.assertRaisesRegex(
            semantic_module.SemanticAlignmentError, "缺少语义文本特征"
        ):
            model._fuse_semantic_and_visual_probs_eval(
                inputs,
                {"logits": visual_logits},
                cached_target_features=target_features,
            )

    def test_eval_fusion_matches_existing_probability_formula(self):
        model = self.make_model(training=False)
        inputs = self.make_episode_inputs()
        target_features = torch.randn(5, 4, 3)
        visual_logits = torch.randn(5, 5)

        with mock.patch.object(semantic_module, "cos_sim", self.fake_cos_sim), \
                mock.patch.object(
                    semantic_module, "OTAM_cum_dist_v2", self.fake_otam
                ):
            result = model._fuse_semantic_and_visual_probs_eval(
                inputs,
                {"logits": visual_logits},
                cached_target_features=target_features,
            )

            flat_targets = target_features.reshape(-1, 3)
            cumulative = []
            for class_name in self.classes:
                stage_features = model._stage_text_features[class_name]
                similarity = self.fake_cos_sim(
                    flat_targets, stage_features
                ) / model.semantic_temperature
                distances = 1 - torch.sigmoid(
                    similarity.reshape(5, 4, stage_features.shape[0])
                )
                cumulative.append(
                    self.fake_otam(
                        distances.transpose(-1, -2).unsqueeze(1)
                    ).squeeze(1)
                )
            semantic_probs = F.softmax(-torch.stack(cumulative, dim=1), dim=1)
            visual_probs = F.softmax(visual_logits, dim=1)
            expected = semantic_probs.pow(0.5) * visual_probs.pow(0.5)

        self.assertTrue(
            torch.allclose(result["logits"], expected, atol=1e-6, rtol=0.0)
        )

    def test_explicit_fallback_is_marked_and_preserves_eval_logits(self):
        model = self.make_model(allow_fallback=True)
        train_result = model._handle_semantic_failure(
            semantic_module.SemanticAlignmentError("forced"),
            phase="train",
            model_dict={"logits": torch.randn(2, 5)},
            reference_tensor=torch.randn(2, 3),
        )
        self.assertEqual(train_result["semantic_alignment_status"], "fallback")
        self.assertEqual(train_result["semantic_alignment_loss"].item(), 0.0)

        model.eval()
        visual_logits = torch.randn(2, 5)
        eval_result = model._handle_semantic_failure(
            semantic_module.SemanticAlignmentError("forced eval"),
            phase="eval",
            model_dict={"logits": visual_logits},
        )
        self.assertEqual(eval_result["semantic_alignment_status"], "fallback")
        self.assertTrue(torch.equal(eval_result["logits"], visual_logits))

    def test_default_failure_policy_raises(self):
        model = self.make_model(allow_fallback=False)
        with self.assertRaisesRegex(
            semantic_module.SemanticAlignmentError, "forced"
        ):
            model._handle_semantic_failure(
                semantic_module.SemanticAlignmentError("forced"),
                phase="train",
                model_dict={"logits": torch.randn(2, 5)},
                reference_tensor=torch.randn(2, 3),
            )


@unittest.skipUnless(train_module is not None, _runtime_skip_reason())
class SemanticTrainerGuardTests(unittest.TestCase):
    @staticmethod
    def make_cfg(allow_fallback=False):
        return SimpleNamespace(
            TRAIN=SimpleNamespace(
                SEMANTIC_LOSS_WEIGHT=1.0,
                ALLOW_SEMANTIC_FALLBACK=allow_fallback,
            )
        )

    def test_active_loss_requires_a_real_graph(self):
        disconnected = torch.tensor(0.0, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "真实计算图"):
            train_module._validate_required_semantic_loss(
                {
                    "semantic_alignment_loss": disconnected,
                    "semantic_alignment_status": "active",
                },
                self.make_cfg(),
            )

        connected = (torch.ones(1, requires_grad=True) * 2.0).sum()
        result = train_module._validate_required_semantic_loss(
            {
                "semantic_alignment_loss": connected,
                "semantic_alignment_status": "active",
            },
            self.make_cfg(),
        )
        self.assertIs(result, connected)

    def test_fallback_is_accepted_only_when_explicitly_enabled(self):
        output = {
            "semantic_alignment_loss": torch.tensor(0.0),
            "semantic_alignment_status": "fallback",
        }
        with self.assertRaisesRegex(RuntimeError, "ALLOW_SEMANTIC_FALLBACK=false"):
            train_module._validate_required_semantic_loss(
                output, self.make_cfg(False)
            )
        result = train_module._validate_required_semantic_loss(
            output, self.make_cfg(True)
        )
        self.assertEqual(result.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
