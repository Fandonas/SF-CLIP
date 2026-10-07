"""纯视觉 OTAM：保留本项目 CNN_OTAM 的路径规则，稳定地计算 FP32 对齐距离。"""

from contextlib import contextmanager
import math

import torch


@contextmanager
def _fp32_context(tensor):
    # 新版接口同时支持 CPU/CUDA；旧版项目仅有 CUDA autocast。
    if hasattr(torch, "autocast"):
        with torch.autocast(device_type=tensor.device.type, enabled=False):
            yield
    elif hasattr(torch.cuda, "amp"):
        with torch.cuda.amp.autocast(enabled=False):
            yield
    else:
        yield


def cos_sim(x, y, epsilon=0.01):
    """与 models/base/few_shot.py 相同：分母为两范数乘积加 epsilon。"""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon 必须为有限正数。")
    with _fp32_context(x):
        x, y = x.float(), y.float()
        numerator = torch.matmul(x, y.transpose(-1, -2))
        denominator = torch.matmul(
            torch.norm(x, dim=-1).unsqueeze(-1),
            torch.norm(y, dim=-1).unsqueeze(-1).transpose(-1, -2),
        ) + epsilon
        return numerator / denominator


def _soft_min(predecessors, lbda):
    return -lbda * torch.logsumexp(torch.stack(predecessors, dim=-1) / -lbda, dim=-1)


def otam_cum_dist(dists, lbda=0.1):
    """单向 OTAM，输入 [查询数, 支持数, 查询帧数, 支持帧数]。

    左右补零、首行累加、首列/末列三路和中间列两路递推与原实现一致。
    用函数式行列表保留梯度，避免原实现 exp 下溢及原地写入。
    """
    if not math.isfinite(lbda) or lbda <= 0:
        raise ValueError("OTAM 平滑参数必须为有限正数。")
    if dists.ndim != 4 or any(size == 0 for size in dists.shape):
        raise ValueError("OTAM 距离必须是非空四维张量。")
    with _fp32_context(dists):
        dists = dists.float()
        query_frames, support_frames = dists.shape[-2:]
        zero = dists.new_zeros(dists.shape[:2])
        previous = [zero]
        for column in range(support_frames):
            previous.append(dists[:, :, 0, column] + previous[-1])
        previous.append(previous[-1])  # 首行右侧补零列。
        for row in range(1, query_frames):
            current = [zero]  # 每行的左侧补零列均为零。
            current.append(
                dists[:, :, row, 0]
                + _soft_min((previous[0], previous[1], current[0]), lbda)
            )
            for column in range(1, support_frames):
                current.append(
                    dists[:, :, row, column]
                    + _soft_min((previous[column], current[-1]), lbda)
                )
            current.append(_soft_min((previous[-2], previous[-1], current[-1]), lbda))
            previous = current
        return previous[-1]


def video_distances(query_features, support_features, lbda=0.1):
    """逐对视频的双向距离；不会先平均多 shot 的帧特征。"""
    if query_features.ndim != 3 or support_features.ndim != 3:
        raise ValueError("视频特征必须为 [视频数, 帧数, 特征维数]。")
    if any(size == 0 for size in query_features.shape + support_features.shape):
        raise ValueError("查询或支持视频特征不能为空。")
    if query_features.shape[-1] != support_features.shape[-1]:
        raise ValueError("查询与支持特征维数不一致。")
    queries, query_frames, dimension = query_features.shape
    supports, support_frames, _ = support_features.shape
    similarities = cos_sim(
        query_features.reshape(-1, dimension), support_features.reshape(-1, dimension)
    )
    distances = (1 - similarities).reshape(queries, query_frames, supports, support_frames)
    distances = distances.permute(0, 2, 1, 3)
    return otam_cum_dist(distances, lbda) + otam_cum_dist(distances.transpose(-1, -2), lbda)
