import os
import re
from typing import Dict, List

import torch

from models.base.few_shot import CNN_OTAM_SF_CLIP, OTAM_cum_dist_v2, cos_sim


class CNN_SEMANTIC_ALIGNMENT_SUPERVISED(CNN_OTAM_SF_CLIP):
    def __init__(self, args):
        super().__init__(args)

        from models.base.few_shot import load

        self.full_clip_model, _ = load(
            args.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=args, jit=False
        )
        self.semantic_stages = self._parse_semantic_stages()
        self.stage_text_features = self._create_stage_text_features()
        self.semantic_loss_weight = getattr(args.TRAIN, "SEMANTIC_LOSS_WEIGHT", 0.5)

    def _parse_semantic_stages(self) -> Dict[str, List[str]]:
        semantic_stages = {}

        for class_name in self.class_real_train:
            stages = re.split(r"[,，]\s*|then\s+", class_name.lower())
            stages = [stage.strip() for stage in stages if stage.strip()]
            semantic_stages[class_name] = stages if len(stages) > 1 else [class_name]

        return semantic_stages

    def _create_stage_text_features(self) -> Dict[str, torch.Tensor]:
        stage_text_features = {}

        from models.base.few_shot import tokenize

        with torch.no_grad():
            for class_name, stages in self.semantic_stages.items():
                stage_features = []
                for stage in stages:
                    stage_text = f"a video of {stage}"
                    text_tokens = tokenize([stage_text]).cuda()
                    stage_feature = self.full_clip_model.encode_text(text_tokens)
                    stage_features.append(stage_feature)

                stage_text_features[class_name] = torch.cat(stage_features, dim=0)

        return stage_text_features

    def _compute_semantic_alignment_loss(
        self, video_features: torch.Tensor, class_labels: torch.Tensor, debug=False
    ) -> torch.Tensor:
        try:
            batch_size, _, _ = video_features.shape
            semantic_losses = []

            for i in range(batch_size):
                class_idx = class_labels[i].item()
                if class_idx < len(self.class_real_train):
                    class_name = self.class_real_train[class_idx]

                    if class_name in self.stage_text_features:
                        stage_text_features = self.stage_text_features[class_name]
                        frame_features = video_features[i]

                        similarity_matrix = cos_sim(frame_features, stage_text_features)

                        if debug and i == 0:
                            print(
                                f"Debug - Similarity matrix range: "
                                f"[{similarity_matrix.min().item():.4f}, "
                                f"{similarity_matrix.max().item():.4f}]"
                            )

                        dists = 1 - similarity_matrix

                        if debug and i == 0:
                            print(
                                f"Debug - Distance matrix range: "
                                f"[{dists.min().item():.4f}, {dists.max().item():.4f}]"
                            )

                        dists_4d = dists.unsqueeze(0).unsqueeze(0)
                        cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)

                        if debug and i == 0:
                            print(f"Debug - OTAM output: {cum_dists[0, 0].item():.4f}")

                        semantic_loss = torch.abs(cum_dists[0, 0])
                        semantic_losses.append(semantic_loss)

            if semantic_losses:
                return torch.stack(semantic_losses).mean()
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)

        except Exception as e:
            print(f"Warning: Semantic alignment loss computation failed: {e}")
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)

    def forward(self, videos, labels):
        batch_size, num_frames, c, h, w = videos.shape
        videos_reshaped = videos.view(batch_size * num_frames, c, h, w)

        video_features, _, _ = self.get_feats(
            videos_reshaped, videos_reshaped, labels
        )
        video_features = video_features.view(batch_size, num_frames, -1)

        feature_classification = self.classification_layer(video_features).mean(1)
        temperature = 0.1

        label_similarities = []
        for i in range(batch_size):
            class_idx = labels[i].item()
            if class_idx < self.text_features_train.shape[0]:
                target_text_feature = self.text_features_train[class_idx : class_idx + 1]
                sample_similarity = (
                    cos_sim(feature_classification[i : i + 1], target_text_feature)
                    / temperature
                )
                label_similarities.append(sample_similarity.squeeze())
            else:
                label_similarities.append(
                    torch.tensor(0.0, device=feature_classification.device)
                )

        text_similarity = torch.stack(label_similarities)
        text_logits = text_similarity

        if hasattr(self, "_debug_text_count") and self._debug_text_count < 3:
            if not hasattr(self, "_debug_text_count"):
                self._debug_text_count = 0
            self._debug_text_count += 1
            print(
                f"Debug - Text similarity range: "
                f"[{text_similarity.min().item():.4f}, {text_similarity.max().item():.4f}]"
            )
            print(f"Debug - Scale value: {self.scale.item():.4f}")
            print(
                f"Debug - Feature classification norm: "
                f"{torch.norm(feature_classification, dim=1).mean().item():.4f}"
            )
            print(
                f"Debug - Text features norm: "
                f"{torch.norm(self.text_features_train, dim=1).mean().item():.4f}"
            )
            print(f"Debug - Current batch labels: {labels[:5].tolist()}")
            print(f"Debug - Text logits shape: {text_logits.shape}")
            print("Debug - Text logits (label similarities for first 5 samples):")
            print(text_logits[:5].detach().cpu().numpy())

            for i in range(min(3, len(labels))):
                label = labels[i].item()
                label_similarity = text_logits[i].item()
                print(
                    f"Debug - Sample {i}, Label {label}, "
                    f"Label similarity: {label_similarity:.4f}"
                )

            print(
                f"Debug - Label similarity input range: "
                f"[{text_logits.min().item():.4f}, {text_logits.max().item():.4f}]"
            )

        debug_semantic = hasattr(self, "_debug_count") and self._debug_count < 3
        if not hasattr(self, "_debug_count"):
            self._debug_count = 0
        if debug_semantic:
            self._debug_count += 1

        semantic_alignment_loss = self._compute_semantic_alignment_loss(
            video_features, labels, debug=debug_semantic
        )

        class_dists = -text_similarity
        return {
            "logits": class_dists,
            "text_logits": text_logits,
            "semantic_alignment_loss": semantic_alignment_loss,
            "video_features": video_features,
            "labels": labels,
        }

    def loss(self, labels, model_output):
        text_similarities = model_output["text_logits"]
        text_similarity_loss = -text_similarities.mean()
        semantic_loss = model_output["semantic_alignment_loss"]
        total_loss = text_similarity_loss + self.semantic_loss_weight * semantic_loss

        return {
            "total_loss": total_loss,
            "text_similarity_loss": text_similarity_loss,
            "semantic_loss": semantic_loss,
            "weighted_semantic_loss": self.semantic_loss_weight * semantic_loss,
        }
