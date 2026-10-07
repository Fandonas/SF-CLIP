"""接收现有 few-shot 任务字典的纯视觉 ViT-OTAM 模型。"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .alignment import video_distances
from .visual import build_visual_encoder


class OTAM_VIT(nn.Module):
    """CLIP ViT-B/16 帧编码 + 双向 OTAM，无文本或额外时序模块。"""

    def __init__(self, cfg, backbone=None):
        super().__init__()
        self.args = cfg
        name = cfg.VIDEO.HEAD.BACKBONE_NAME
        if name != "ViT-B/16":
            raise ValueError("OTAM_VIT 仅支持 CLIP ViT-B/16，实际为 {}。".format(name))
        self.num_frames = int(cfg.DATA.NUM_INPUT_FRAMES)
        if self.num_frames <= 0:
            raise ValueError("DATA.NUM_INPUT_FRAMES 必须为正数。")
        self.lbda = float(getattr(cfg.VIDEO.HEAD, "OTAM_LAMBDA", 0.1))
        if not math.isfinite(self.lbda) or self.lbda <= 0:
            raise ValueError("VIDEO.HEAD.OTAM_LAMBDA 必须为有限正数。")
        self.freeze_backbone = bool(getattr(cfg.TRAIN, "FREEZE_BACKBONE", False))
        if self.freeze_backbone and cfg.TRAIN.ENABLE:
            raise ValueError("纯 OTAM 只有视觉骨干参数；训练时不能冻结整个骨干。")
        # 注入骨干时完全跳过权重解析、下载和生产模型构造，供轻量测试使用。
        self.backbone = backbone if backbone is not None else build_visual_encoder(
            getattr(cfg.VIDEO.HEAD, "PRETRAINED_PATH", "")
        )
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _encode_frames(self, images, name):
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("{} 必须为 [视频数 * 帧数, 3, 高, 宽] 张量。".format(name))
        if images.shape[0] == 0 or images.shape[0] % self.num_frames:
            raise ValueError("{} 的帧数必须是 NUM_INPUT_FRAMES 的正整数倍。".format(name))
        resolution = getattr(self.backbone, "input_resolution", None)
        if resolution is not None and tuple(images.shape[-2:]) != (resolution, resolution):
            raise ValueError("{} 的输入分辨率必须为 {}×{}。".format(name, resolution, resolution))
        features = self.backbone(images)
        if features.ndim != 2 or features.shape[0] != images.shape[0]:
            raise ValueError("视觉骨干必须为每帧返回一个一维特征。")
        return features.reshape(-1, self.num_frames, features.shape[-1])

    def forward(self, inputs):
        support = self._encode_frames(inputs["support_set"], "support_set")
        query = self._encode_frames(inputs["target_set"], "target_set")
        labels = inputs["support_labels"]
        if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.numel() != support.shape[0]:
            raise ValueError("support_labels 必须为每条支持视频提供一个类别标签。")
        labels = labels.to(device=support.device)
        if not torch.equal(labels, labels.long()):
            raise ValueError("support_labels 必须使用整数类别编号。")
        labels = labels.long()
        classes = torch.unique(labels, sorted=True)
        distances = video_distances(query, support, self.lbda)
        class_distances = torch.stack(
            [distances[:, labels == label].mean(dim=1) for label in classes], dim=1
        )
        return {"logits": -class_distances}

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
