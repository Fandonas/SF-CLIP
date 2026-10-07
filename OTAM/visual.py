# Copyright (c) 2021 OpenAI
# 视觉类取自 OpenAI CLIP 的固定提交，保留其 MIT 许可证：
# d05afc436d78f1c48dc0dbf8e5980a9d471f35f6 / clip/model.py
# 许可证见 third_party/openai_clip/LICENSE；下方权重加载辅助函数为本项目实现。
"""仅包含 CLIP 视觉编码器及其预训练权重加载，不构造文本模型。"""

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import hashlib
import os
import tempfile
import urllib.request
import warnings

import torch
from torch import nn


CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_VIT_B16_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_VIT_B16_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    + CLIP_VIT_B16_SHA256 + "/ViT-B-16.pt"
)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x



def _sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pretrained_path(pretrained_path="", cache_dir=None):
    """本地路径直接使用；默认缓存必须通过官方 SHA-256 校验。"""
    if pretrained_path:
        path = Path(pretrained_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("视觉预训练权重文件不存在: {}".format(path))
        return path

    root = Path(cache_dir).expanduser() if cache_dir is not None else Path.home() / ".cache" / "clip"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "ViT-B-16.pt"
    if destination.is_file():
        if _sha256_file(destination) == CLIP_VIT_B16_SHA256:
            return destination
        warnings.warn("CLIP ViT-B/16 缓存的 SHA-256 不匹配，将重新下载。")
    elif destination.exists():
        raise RuntimeError("权重缓存路径不是文件: {}".format(destination))

    # 每个进程使用独立临时文件；仅在下载完整并校验后替换公共缓存。
    descriptor, temporary = tempfile.mkstemp(prefix="ViT-B-16-", suffix=".part", dir=str(root))
    os.close(descriptor)
    try:
        with urllib.request.urlopen(CLIP_VIT_B16_URL, timeout=60) as response:
            with open(temporary, "wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
        if _sha256_file(temporary) != CLIP_VIT_B16_SHA256:
            raise RuntimeError("下载的 CLIP ViT-B/16 权重 SHA-256 校验失败。")
        # 另一个训练进程可能已经填充了同一缓存。
        if destination.is_file() and _sha256_file(destination) == CLIP_VIT_B16_SHA256:
            return destination
        try:
            os.replace(temporary, str(destination))
        except PermissionError:
            # Windows 上另一个进程可能正在读取刚刚完成的有效缓存。
            if not destination.is_file() or _sha256_file(destination) != CLIP_VIT_B16_SHA256:
                raise
        return destination
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def extract_visual_state_dict(checkpoint):
    """从完整 CLIP state_dict 或视觉 state_dict 提取视觉权重。"""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("视觉 checkpoint 必须是 state_dict 或包含 state_dict 的字典。")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise ValueError("视觉 state_dict 必须使用字符串键。")
    if any(key.startswith("visual.") for key in state):
        visual = OrderedDict(
            (key[len("visual."):], value)
            for key, value in state.items() if key.startswith("visual.")
        )
    elif "conv1.weight" in state:
        visual = OrderedDict(state.items())
    else:
        raise ValueError("checkpoint 中未找到 visual.* 或视觉编码器权重。")
    if not all(isinstance(value, torch.Tensor) for value in visual.values()):
        raise ValueError("视觉 state_dict 中包含非张量权重。")
    return visual


def read_visual_checkpoint(path):
    """读取官方 TorchScript 文件或普通 state_dict，始终映射到 CPU。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("视觉预训练权重文件不存在: {}".format(path))
    try:
        checkpoint = torch.jit.load(str(path), map_location="cpu")
    except RuntimeError:
        try:
            # 新版 PyTorch 的安全 state_dict 加载接口。
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:
            # 兼容项目原有的旧版 PyTorch。
            checkpoint = torch.load(str(path), map_location="cpu")
        return extract_visual_state_dict(checkpoint)
    return extract_visual_state_dict(checkpoint.state_dict())


def load_visual_state_dict(visual, checkpoint):
    """严格加载：缺失键、多余视觉键或形状不匹配均明确报错。"""
    state = extract_visual_state_dict(checkpoint)
    try:
        visual.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError("视觉权重不匹配（严格加载）: {}".format(error)) from error
    return visual


def build_visual_encoder(pretrained_path="", cache_dir=None):
    """构建带 512 维输出投影的 CLIP ViT-B/16，权重及参数采用 FP32。"""
    path = resolve_pretrained_path(pretrained_path, cache_dir)
    state = read_visual_checkpoint(path)
    expected_shapes = {
        "conv1.weight": (768, 3, 16, 16),
        "positional_embedding": (197, 768),
        "proj": (768, 512),
    }
    for key, expected in expected_shapes.items():
        value = state.get(key)
        actual = tuple(value.shape) if value is not None else None
        if actual != expected:
            raise RuntimeError(
                "ViT-B/16 视觉权重不匹配: {} 期望 {}，实际 {}。".format(key, expected, actual)
            )
    visual = VisionTransformer(
        input_resolution=224, patch_size=16, width=768,
        layers=12, heads=12, output_dim=512,
    ).float()
    return load_visual_state_dict(visual, state)
