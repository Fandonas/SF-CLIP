import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import sys
import os

# 使用原始项目的模块
from models.base.few_shot import CNN_OTAM_SF_CLIP, cos_sim, OTAM_cum_dist_v2
from models.base.few_shot import HEAD_REGISTRY
from einops import rearrange
from models.base.few_shot import extract_class_indices


class SemanticAlignmentError(RuntimeError):
    """语义对齐启用后无法可靠完成时抛出的内部异常。"""


@HEAD_REGISTRY.register()
class CNN_SEMANTIC_ALIGNMENT_FEW_SHOT(CNN_OTAM_SF_CLIP):
    """
    小样本学习的语义对齐模型

    核心功能：
    1. 帧对齐：使用OTAM算法进行support和query视频的帧级别对齐
    2. 语义对齐：将视频帧与动作语义阶段进行对齐
    3. 损失函数构成：帧对齐损失（few-shot分类损失） + 语义对齐损失（OTAM算法）
    4. 保持few-shot学习框架，使用support和target划分

    语义对齐机制：
    - 将动作类别名称解析为多个语义阶段
    - 使用CLIP对每个语义阶段进行文本编码
    - 直接使用每一帧特征与语义阶段进行OTAM对齐
    - 找到帧与语义阶段的最优对齐路径

    特点：不使用类别级别的文本对齐，只保留帧对齐和语义对齐
    """

    def __init__(self, cfg):
        super(CNN_SEMANTIC_ALIGNMENT_FEW_SHOT, self).__init__(cfg)

        # 语义对齐损失权重
        self.semantic_loss_weight = getattr(cfg.TRAIN, 'SEMANTIC_LOSS_WEIGHT', 0.5)

        # 温度参数：用于控制语义对齐相似度的尺度（类似全监督模型）
        # 较小的temperature使相似度分布更加sharp，较大的使分布更加smooth
        self.semantic_temperature = getattr(cfg.TRAIN, 'SEMANTIC_TEMPERATURE', 0.1)
        # True→原始行为(行=帧)；False→转置行为(行=阶段，一阶段多帧)
        self.semantic_transpose = getattr(cfg.TRAIN, 'SEMANTIC_TRANSPOSE', False)
        self.allow_semantic_fallback = bool(
            getattr(cfg.TRAIN, 'ALLOW_SEMANTIC_FALLBACK', False)
        )

        semantic_combine = bool(getattr(cfg.TRAIN, 'SEMANTIC_COMBINE', False))
        if not np.isfinite(self.semantic_loss_weight) or self.semantic_loss_weight < 0:
            raise SemanticAlignmentError(
                f"SEMANTIC_LOSS_WEIGHT 必须是有限的非负数，当前值为 {self.semantic_loss_weight!r}"
            )
        if (
            (self.semantic_loss_weight > 0 or semantic_combine)
            and (
                not np.isfinite(self.semantic_temperature)
                or self.semantic_temperature <= 0
            )
        ):
            raise SemanticAlignmentError(
                f"SEMANTIC_TEMPERATURE 必须是有限正数，当前值为 {self.semantic_temperature!r}"
            )

        # Query Embedding 控制参数
        self.use_query_embed_train = getattr(cfg.TRAIN, 'USE_QUERY_EMBED_TRAIN', True)
        self.use_query_embed_eval = getattr(cfg.TRAIN, 'USE_QUERY_EMBED_EVAL', False)

        # Mid Layer 控制参数
        self.use_mid_layer_train = getattr(cfg.TRAIN, 'USE_MID_LAYER_TRAIN', True)
        self.use_mid_layer_eval = getattr(cfg.TRAIN, 'USE_MID_LAYER_EVAL', False)
        self.use_mid_layer2 = getattr(cfg.TRAIN, 'USE_MID_LAYER2', False)
        self.use_mid_layer2_semantic_train = getattr(cfg.TRAIN, 'USE_MID_LAYER2_SEMANTIC_TRAIN', False)
        self.use_mid_layer2_semantic_eval = getattr(cfg.TRAIN, 'USE_MID_LAYER2_SEMANTIC_EVAL', False)
        self.use_mid_layer_relu = getattr(cfg.TRAIN, 'USE_MID_LAYER_RELU', False)

        # 延迟初始化：在第一次使用时才创建文本特征（避免初始化时CUDA不可用）
        self._full_clip_model = None
        self._semantic_stages = None
        self._stage_text_features = None
        self._text_features_initialized = False
        self._semantic_initialization_error = None
        self._semantic_fallback_warnings = set()

        # 可学习的训练类 query_embed：用冻结的 CLIP 文本特征初始化，训练时随梯度更新
        # 形状 [num_train_classes, mid_dim]，参数量极小（如64×512=32K）
        # 测试阶段仍使用冻结的 text_features_test（novel类无法预训练）
        num_train_classes = self.text_features_train.shape[0]
        if self.use_query_embed_train:
            # 训练时使用可学习 Embedding：参与梯度更新，DDP 正常追踪
            # use_query_embed_eval=True 依赖此模块，须与 use_query_embed_train 同时开启
            self.query_embed_train = nn.Embedding(num_train_classes, self.mid_dim)
            self.query_embed_train.weight = nn.Parameter(
                self.text_features_train.detach().clone()
            )
        # use_query_embed_train=False：不创建此模块
        # - 训练时走 text_features_train（frozen buffer），DDP 不追踪，无未使用参数问题
        # - 评估时走 text_features_test，query_embed_train 不存在也不会被访问
        # - 若误将 use_query_embed_eval=True 而 use_query_embed_train=False，
        #   会在 eval forward 抛出 AttributeError，提示配置错误

        # 仅在训练或评估任一路径需要时才覆盖父类的空 nn.Sequential()
        # 若两者均为 False，保持父类 mid_layer = nn.Sequential()（0参数），DDP 不追踪
        if self.use_mid_layer_train or self.use_mid_layer_eval:
            if self.use_mid_layer_relu:
                self.mid_layer = nn.Sequential(
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.ReLU()
                )
            else:
                self.mid_layer = nn.Linear(self.mid_dim, self.mid_dim)

        # 创建 mid_layer2：与 mid_layer 并行处理 context_support
        if self.use_mid_layer2:
            if self.use_mid_layer_relu:
                self.mid_layer2 = nn.Sequential(
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.ReLU()
                )
            else:
                self.mid_layer2 = nn.Linear(self.mid_dim, self.mid_dim)

    @staticmethod
    def _require_finite_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
        """验证语义路径中的张量非空且全部为有限值。"""
        if not isinstance(tensor, torch.Tensor):
            raise SemanticAlignmentError(
                f"{name} 必须是 torch.Tensor，实际为 {type(tensor).__name__}"
            )
        if tensor.numel() == 0:
            raise SemanticAlignmentError(f"{name} 不能为空张量")
        if not torch.isfinite(tensor).all().item():
            raise SemanticAlignmentError(
                f"{name} 包含 NaN 或 Inf；shape={tuple(tensor.shape)}, "
                f"dtype={tensor.dtype}, device={tensor.device}"
            )
        return tensor

    def _handle_semantic_failure(
        self,
        exc: Exception,
        phase: str,
        model_dict: Dict[str, torch.Tensor],
        reference_tensor: Optional[torch.Tensor] = None,
    ):
        """在唯一边界执行 fail-fast 或用户显式允许的帧对齐回退。"""
        if isinstance(exc, SemanticAlignmentError):
            semantic_error = exc
        else:
            semantic_error = SemanticAlignmentError(
                f"{phase} 阶段语义对齐失败: {exc}"
            )

        if not self.allow_semantic_fallback:
            if semantic_error is exc:
                raise semantic_error
            raise semantic_error from exc

        warning_key = (phase, str(semantic_error))
        if warning_key not in self._semantic_fallback_warnings:
            print(
                "Warning: ALLOW_SEMANTIC_FALLBACK=true，"
                f"{phase} 阶段显式回退到纯帧对齐: {semantic_error}"
            )
            self._semantic_fallback_warnings.add(warning_key)

        fallback_dict = dict(model_dict)
        fallback_dict['semantic_alignment_status'] = 'fallback'
        if phase == 'train':
            if reference_tensor is None:
                raise SemanticAlignmentError(
                    "训练语义回退缺少用于构造零损失的参考张量"
                )
            fallback_dict['semantic_alignment_loss'] = reference_tensor.new_zeros(())
        return fallback_dict

    def _get_class_name(self, label_idx: int) -> str:
        """
        根据label_idx和当前训练状态获取正确的类别名称

        重要说明：
        - 训练集和测试集的ID空间是独立的，都从0开始
        - 训练时使用 class_real_train 的实际长度校验 label_idx
        - 测试时使用 class_real_test 的实际长度校验 label_idx

        Args:
            label_idx: 类别索引（来自real_support_labels或real_target_labels）

        Returns:
            对应的类别名称字符串
        """
        label_idx = int(label_idx)

        # 根据模型的training状态选择对应的类别列表
        if self.training:
            # 训练模式：使用训练集类别列表
            if 0 <= label_idx < len(self.class_real_train):
                return self.class_real_train[label_idx]
            raise SemanticAlignmentError(
                f"训练真实类别 ID {label_idx} 越界；合法范围为 "
                f"[0, {len(self.class_real_train) - 1}]"
            )
        else:
            # 评估/测试模式：使用测试集类别列表
            if 0 <= label_idx < len(self.class_real_test):
                return self.class_real_test[label_idx]
            raise SemanticAlignmentError(
                f"评估真实类别 ID {label_idx} 越界；合法范围为 "
                f"[0, {len(self.class_real_test) - 1}]"
            )

    def _build_semantic_episode_class_names(
        self,
        inputs,
        support_bs: int,
        target_bs: int,
    ) -> Tuple[List[str], List[str]]:
        """构建并严格验证 support/target 的 episode 类别映射。"""
        support_labels = inputs['support_labels']
        support_real_labels = inputs['real_support_labels']

        if len(support_labels) != support_bs or len(support_real_labels) != support_bs:
            raise SemanticAlignmentError(
                "support 标签数量与 support 特征数量不一致："
                f"episodic={len(support_labels)}, real={len(support_real_labels)}, "
                f"features={support_bs}"
            )
        if 'real_target_labels' not in inputs:
            raise SemanticAlignmentError(
                "语义对齐启用时必须提供 real_target_labels，禁止使用 support 标签补齐"
            )
        target_real_labels = inputs['real_target_labels']
        if len(target_real_labels) != target_bs:
            raise SemanticAlignmentError(
                "real_target_labels 数量与 target 特征数量不一致："
                f"labels={len(target_real_labels)}, features={target_bs}"
            )

        unique_labels = torch.unique(support_labels)
        expected_way = int(
            getattr(
                self.args.TRAIN,
                'WAY' if self.training else 'WAT_TEST',
                getattr(self.args.TRAIN, 'WAY', len(unique_labels)),
            )
        )
        if len(unique_labels) != expected_way:
            raise SemanticAlignmentError(
                f"episode 类别数必须为 {expected_way}，实际为 {len(unique_labels)}"
            )

        episode_class_names: List[str] = []
        support_class_names: List[str] = []
        for episodic_label in unique_labels:
            class_indices = extract_class_indices(support_labels, episodic_label)
            real_ids = {
                int(support_real_labels[index].item())
                for index in class_indices
            }
            if len(real_ids) != 1:
                raise SemanticAlignmentError(
                    f"episodic label {int(episodic_label.item())} 对应了多个真实类别 ID: "
                    f"{sorted(real_ids)}"
                )
            episode_class_names.append(self._get_class_name(next(iter(real_ids))))

        if len(set(episode_class_names)) != expected_way:
            raise SemanticAlignmentError(
                "多个 episodic label 映射到了同一个真实类别："
                f"{episode_class_names}"
            )

        for real_label in support_real_labels:
            support_class_names.append(self._get_class_name(int(real_label.item())))
        target_class_names = [
            self._get_class_name(int(real_label.item()))
            for real_label in target_real_labels
        ]
        unknown_targets = sorted(set(target_class_names) - set(episode_class_names))
        if unknown_targets:
            raise SemanticAlignmentError(
                "target 中出现了不属于 support episode 的真实类别："
                f"{unknown_targets}"
            )

        return support_class_names + target_class_names, episode_class_names

    def _ensure_text_features_initialized(self):
        """确保文本特征已初始化（延迟初始化）"""
        if self._text_features_initialized:
            return

        if self._semantic_initialization_error is not None:
            raise SemanticAlignmentError(
                "语义文本特征此前已初始化失败，本进程不再重复加载："
                f"{self._semantic_initialization_error}"
            )

        try:
            # 获取完整的CLIP模型用于文本编码
            from models.base.few_shot import load

            # 确定设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._full_clip_model, _ = load(self.args.VIDEO.HEAD.BACKBONE_NAME, device=device, cfg=self.args, jit=False)

            # 语义阶段解析
            self._semantic_stages = self._parse_semantic_stages()

            all_classes = self.class_real_train + self.class_real_test
            if len(set(all_classes)) != len(all_classes):
                raise SemanticAlignmentError(
                    "训练/测试 CLASS_NAME 中存在重复类别，无法建立一一对应的语义特征"
                )
            expected_classes = set(all_classes)
            if set(self._semantic_stages) != expected_classes:
                missing = sorted(expected_classes - set(self._semantic_stages))
                extra = sorted(set(self._semantic_stages) - expected_classes)
                raise SemanticAlignmentError(
                    f"语义阶段解析类别集合不完整；missing={missing}, extra={extra}"
                )

            # 为每个阶段创建文本特征
            stage_text_features = self._create_stage_text_features()
            if set(stage_text_features) != expected_classes:
                missing = sorted(expected_classes - set(stage_text_features))
                extra = sorted(set(stage_text_features) - expected_classes)
                raise SemanticAlignmentError(
                    f"语义文本特征类别集合不完整；missing={missing}, extra={extra}"
                )
            for class_name, features in stage_text_features.items():
                self._require_finite_tensor(
                    f"类别 '{class_name}' 的语义文本特征", features
                )
                if features.ndim != 2 or features.shape[1] != self.mid_dim:
                    raise SemanticAlignmentError(
                        f"类别 '{class_name}' 的语义文本特征维度错误："
                        f"shape={tuple(features.shape)}, expected_dim={self.mid_dim}"
                    )

            self._stage_text_features = stage_text_features

            self._text_features_initialized = True
        except Exception as e:
            if isinstance(e, SemanticAlignmentError):
                semantic_error = e
            else:
                semantic_error = SemanticAlignmentError(
                    f"语义文本特征初始化失败: {e}"
                )
            self._semantic_stages = None
            self._stage_text_features = None
            self._text_features_initialized = False
            # 只缓存字符串，避免异常 traceback 间接持有已释放的完整 CLIP 模型。
            self._semantic_initialization_error = str(semantic_error)
            if semantic_error is e:
                raise semantic_error
            raise semantic_error from e
        finally:
            # 成功或失败都释放临时完整 CLIP；不改变其加载时机和显存方案。
            self._full_clip_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _parse_semantic_stages(self) -> Dict[str, List[str]]:  # 分割阶段"," ", " ", then "
        """解析类别名称的语义阶段"""
        semantic_stages = {}

        # 合并训练和测试类别（few-shot中需要）
        all_classes = self.class_real_train + self.class_real_test

        for class_name in all_classes:
            # 使用逗号和"then"分割语义阶段
            stages = re.split(r'[,，]\s*|then\s+', class_name.lower())
            stages = [stage.strip() for stage in stages if stage.strip()]

            if len(stages) > 1:
                semantic_stages[class_name] = stages
            else:
                # 如果没有明确的分割，保持原始名称
                semantic_stages[class_name] = [class_name]

        return semantic_stages

    def _create_stage_text_features(self) -> Dict[str, torch.Tensor]:  # 分阶段提取特征后cat一起
        """为每个类别的语义阶段创建文本特征"""
        stage_text_features = {}

        # 导入tokenize函数
        from models.base.few_shot import tokenize

        # 确定设备
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 使用torch.no_grad()避免梯度计算问题
        with torch.no_grad():
            for class_name, stages in self._semantic_stages.items():
                stage_features = []
                for stage in stages:
                    try:
                        # 使用CLIP编码每个阶段的文本
                        stage_text = f"a video of {stage}"
                        # 使用完整的CLIP模型进行文本编码
                        text_tokens = tokenize([stage_text]).to(device)
                        stage_feature = self._full_clip_model.encode_text(text_tokens)
                    except Exception as e:
                        raise SemanticAlignmentError(
                            f"类别 '{class_name}' 的阶段 '{stage}' 文本编码失败: {e}"
                        ) from e

                    self._require_finite_tensor(
                        f"类别 '{class_name}' 的阶段 '{stage}' 文本特征",
                        stage_feature,
                    )
                    if (
                        stage_feature.ndim != 2
                        or stage_feature.shape[0] != 1
                        or stage_feature.shape[1] != self.mid_dim
                    ):
                        raise SemanticAlignmentError(
                            f"类别 '{class_name}' 的阶段 '{stage}' 文本特征维度错误："
                            f"shape={tuple(stage_feature.shape)}, expected=(1, {self.mid_dim})"
                        )
                    stage_features.append(stage_feature)

                if len(stage_features) != len(stages) or not stage_features:
                    raise SemanticAlignmentError(
                        f"类别 '{class_name}' 的语义阶段不完整："
                        f"expected={len(stages)}, actual={len(stage_features)}"
                    )
                # 堆叠所有阶段特征 [num_stages, feature_dim]
                stage_text_features[class_name] = torch.cat(stage_features, dim=0)

        return stage_text_features

    def _compute_semantic_alignment_loss(self, video_features: torch.Tensor,
                                         class_names: List[str], debug=False) -> torch.Tensor:
        """
        计算语义对齐损失，使用OTAM算法进行帧级别的语义阶段对齐

        核心改进：
        1. 不对视频帧进行分组，直接使用每一帧
        2. 每一帧都能与具体的语义阶段进行对齐
        3. 使用OTAM算法找到帧与语义阶段的最优对齐路径

        Args:
            video_features: [batch_size, num_frames, feature_dim] 视频特征
            class_names: [batch_size] 类别名称列表

        Returns:
            semantic_loss: 语义对齐损失
        """
        self._ensure_text_features_initialized()
        self._require_finite_tensor("语义视频特征", video_features)
        if video_features.ndim != 3:
            raise SemanticAlignmentError(
                f"语义视频特征必须为 [batch, frames, dim]，实际为 {tuple(video_features.shape)}"
            )

        batch_size, _, feature_dim = video_features.shape
        if len(class_names) != batch_size:
            raise SemanticAlignmentError(
                f"类别名称数量与视频数量不一致：names={len(class_names)}, batch={batch_size}"
            )

        missing_classes = sorted(set(class_names) - set(self._stage_text_features))
        if missing_classes:
            raise SemanticAlignmentError(
                f"以下类别缺少语义文本特征：{missing_classes}"
            )

        semantic_losses = []
        for i, class_name in enumerate(class_names):
            stage_text_features = self._stage_text_features[class_name].to(
                video_features.device
            )
            self._require_finite_tensor(
                f"类别 '{class_name}' 的语义文本特征", stage_text_features
            )
            if stage_text_features.ndim != 2 or stage_text_features.shape[1] != feature_dim:
                raise SemanticAlignmentError(
                    f"类别 '{class_name}' 的文本/视频特征维度不一致："
                    f"text={tuple(stage_text_features.shape)}, video_dim={feature_dim}"
                )

            frame_features = video_features[i]
            similarity_matrix = cos_sim(
                frame_features, stage_text_features
            ) / self.semantic_temperature
            self._require_finite_tensor(
                f"类别 '{class_name}' 的温度缩放相似度", similarity_matrix
            )

            if debug and i == 0:
                print(
                    "Debug - Similarity matrix range (after temperature): "
                    f"[{similarity_matrix.min().item():.4f}, "
                    f"{similarity_matrix.max().item():.4f}]"
                )
                print(f"Debug - Semantic temperature: {self.semantic_temperature}")

            dists = 1 - torch.sigmoid(similarity_matrix)
            self._require_finite_tensor(f"类别 '{class_name}' 的语义距离", dists)
            if not self.semantic_transpose:
                dists_4d = dists.T.unsqueeze(0).unsqueeze(0)
            else:
                dists_4d = dists.unsqueeze(0).unsqueeze(0)
            cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)
            self._require_finite_tensor(
                f"类别 '{class_name}' 的 OTAM 累积距离", cum_dists
            )

            if debug and i == 0:
                print(f"Debug - OTAM output: {cum_dists[0, 0].item():.4f}")
            semantic_losses.append(torch.abs(cum_dists[0, 0]))

        if not semantic_losses:
            raise SemanticAlignmentError("语义损失没有任何有效样本")
        loss = torch.stack(semantic_losses).mean()
        self._require_finite_tensor("语义对齐损失", loss)
        return loss

    def _compute_semantic_alignment_loss_multi_class(self,
                                                     video_features: torch.Tensor,
                                                     class_names: List[str],
                                                     debug: bool = False) -> torch.Tensor:
        """
        多类别语义对齐版本（向量化实现）：
        - 对于每个视频，与当前 episode 中出现的每一个标签的语义阶段序列做 OTAM 对齐；
        - 得到按类别的累积距离向量 cum_dists[class]；
        - 使用 -cum_dists 作为 logits，经 softmax / CrossEntropy 与真实标签构成语义分类损失；
        - 保留 temperature 缩放和 sigmoid 归一化：cos_sim / T → sigmoid → 1 - sigmoid。

        向量化优化：将原来的 batch_size × num_classes 双重循环改为按类别逐次批量处理，
        OTAM 调用次数从 batch_size×num_classes 次降至 num_classes 次，
        数学结果与原始逐个循环完全等价。

        Args:
            video_features: [batch_size, num_frames, feature_dim]，support 和 target 特征拼接
            class_names:    [batch_size]，与 video_features 一一对应的真实类别名称
        """
        self._ensure_text_features_initialized()
        self._require_finite_tensor("多类别语义视频特征", video_features)
        if video_features.ndim != 3:
            raise SemanticAlignmentError(
                f"语义视频特征必须为 [batch, frames, dim]，实际为 {tuple(video_features.shape)}"
            )

        device = video_features.device
        batch_size, num_frames, feature_dim = video_features.shape
        if len(class_names) != batch_size:
            raise SemanticAlignmentError(
                f"类别名称数量与视频数量不一致：names={len(class_names)}, batch={batch_size}"
            )

        episode_classes: List[str] = []
        for name in class_names:
            if not isinstance(name, str) or not name:
                raise SemanticAlignmentError(f"episode 包含无效类别名称：{name!r}")
            if name not in episode_classes:
                episode_classes.append(name)

        expected_way = int(
            getattr(
                self.args.TRAIN,
                'WAY' if self.training else 'WAT_TEST',
                getattr(self.args.TRAIN, 'WAY', len(episode_classes)),
            )
        )
        if len(episode_classes) != expected_way:
            raise SemanticAlignmentError(
                f"语义分类必须保持 {expected_way}-way，实际为 {len(episode_classes)}-way"
            )

        missing_classes = sorted(set(episode_classes) - set(self._stage_text_features))
        if missing_classes:
            raise SemanticAlignmentError(
                f"episode 类别缺少语义文本特征：{missing_classes}"
            )

        episode_stage_feats = {}
        for class_name in episode_classes:
            features = self._stage_text_features[class_name].to(device)
            self._require_finite_tensor(
                f"类别 '{class_name}' 的语义文本特征", features
            )
            if features.ndim != 2 or features.shape[1] != feature_dim:
                raise SemanticAlignmentError(
                    f"类别 '{class_name}' 的文本/视频特征维度不一致："
                    f"text={tuple(features.shape)}, video_dim={feature_dim}"
                )
            episode_stage_feats[class_name] = features

        frames_flat = video_features.reshape(batch_size * num_frames, feature_dim)
        all_cum_dists_list: List[torch.Tensor] = []
        for cls_idx, class_name in enumerate(episode_classes):
            stage_text_features = episode_stage_feats[class_name]
            num_stages = stage_text_features.shape[0]
            sim_flat = cos_sim(
                frames_flat, stage_text_features
            ) / self.semantic_temperature
            self._require_finite_tensor(
                f"类别 '{class_name}' 的温度缩放相似度", sim_flat
            )
            similarity_matrix = sim_flat.reshape(
                batch_size, num_frames, num_stages
            )

            if debug and cls_idx == 0:
                sm0 = similarity_matrix[0]
                print(
                    "Debug(multi) - Similarity matrix range (after temperature): "
                    f"[{sm0.min().item():.4f}, {sm0.max().item():.4f}]"
                )
                print(f"Debug(multi) - Semantic temperature: {self.semantic_temperature}")

            dists = 1 - torch.sigmoid(similarity_matrix)
            self._require_finite_tensor(
                f"类别 '{class_name}' 的语义距离", dists
            )
            if not self.semantic_transpose:
                dists_4d = dists.transpose(-1, -2).unsqueeze(1)
            else:
                dists_4d = dists.unsqueeze(1)
            cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)
            self._require_finite_tensor(
                f"类别 '{class_name}' 的 OTAM 累积距离", cum_dists
            )
            if cum_dists.shape != (batch_size, 1):
                raise SemanticAlignmentError(
                    f"类别 '{class_name}' 的 OTAM 输出维度错误："
                    f"shape={tuple(cum_dists.shape)}, expected=({batch_size}, 1)"
                )

            if debug and cls_idx == 0:
                print(
                    "Debug(multi) - OTAM output (first video, first class): "
                    f"{cum_dists[0, 0].item():.4f}"
                )
            all_cum_dists_list.append(cum_dists.squeeze(1))

        all_cum_dists = torch.stack(all_cum_dists_list, dim=1)
        self._require_finite_tensor("多类别语义 OTAM 距离", all_cum_dists)
        logits_all = -all_cum_dists
        self._require_finite_tensor("多类别语义 logits", logits_all)
        gt_indices = torch.tensor(
            [episode_classes.index(name) for name in class_names],
            device=device,
            dtype=torch.long,
        )
        loss = F.cross_entropy(logits_all, gt_indices)
        self._require_finite_tensor("多类别语义对齐损失", loss)
        if loss.numel() != 1:
            raise SemanticAlignmentError(
                f"语义对齐损失必须是标量，实际 shape={tuple(loss.shape)}"
            )
        if self.training and self.semantic_loss_weight > 0 and loss.grad_fn is None:
            raise SemanticAlignmentError(
                "语义对齐损失没有 grad_fn，无法真实参与反向传播"
            )
        return loss

    def _fuse_semantic_and_visual_probs_eval(self, inputs, model_dict, cached_target_features=None):
        """
        仅在评估模式下调用：
        使用"语义对齐概率 + 帧对齐概率"作为最终分类结果，
        融合方式模仿文本对齐 + 帧对齐的 COMBINE 分支：

            p_fused ∝ p_semantic^β * p_visual^(1-β)
            logits = -p_fused

        其中：
        - p_visual 来自当前 model_dict["logits"] 的 softmax（few-shot OTAM 帧对齐）
        - p_semantic 来自"目标视频 vs episode 内每个标签的语义阶段序列"的 OTAM 对齐，
          概率计算方式参照文本对齐：对 -cum_dists 做 softmax。

        参与概率计算的视频与文本对齐一致：只使用当前 episode 的 query/target 视频。
        """
        # 只在测试模式并且显式打开 SEMANTIC_COMBINE 时生效
        if self.training:
            return model_dict
        if not hasattr(self.args.TRAIN, "SEMANTIC_COMBINE") or not self.args.TRAIN.SEMANTIC_COMBINE:
            return model_dict
        if "logits" not in model_dict:
            raise SemanticAlignmentError("评估语义融合缺少视觉 logits")

        support_images = inputs["support_set"]
        target_images = inputs["target_set"]
        support_real_class = inputs["real_support_labels"]

        # 优先使用预计算的目标特征，避免重复调用 get_feats（backbone 前向）
        if cached_target_features is not None:
            target_features = cached_target_features
        else:
            _, target_features, _ = self.get_feats(
                support_images, target_images, support_real_class
            )

        self._require_finite_tensor("评估目标视频特征", target_features)
        if target_features.ndim != 3:
            raise SemanticAlignmentError(
                f"评估目标特征必须为 [batch, frames, dim]，实际为 {tuple(target_features.shape)}"
            )
        device = target_features.device
        target_bs, num_frames, feature_dim = target_features.shape
        if target_bs == 0:
            raise SemanticAlignmentError("评估 episode 不包含 target 视频")

        _, episode_class_names = self._build_semantic_episode_class_names(
            inputs, len(inputs["support_labels"]), target_bs
        )
        num_classes = len(episode_class_names)

        visual_logits = model_dict["logits"]
        self._require_finite_tensor("评估帧对齐 logits", visual_logits)
        expected_shape = (target_bs, num_classes)
        if tuple(visual_logits.shape) != expected_shape:
            raise SemanticAlignmentError(
                f"评估帧对齐 logits 维度错误：shape={tuple(visual_logits.shape)}, "
                f"expected={expected_shape}"
            )
        visual_probs = F.softmax(visual_logits, dim=1)
        self._require_finite_tensor("评估帧对齐概率", visual_probs)

        self._ensure_text_features_initialized()
        missing_classes = sorted(
            set(episode_class_names) - set(self._stage_text_features)
        )
        if missing_classes:
            raise SemanticAlignmentError(
                f"评估 episode 类别缺少语义文本特征：{missing_classes}"
            )

        frames_flat = target_features.reshape(target_bs * num_frames, feature_dim)
        semantic_cum_list: List[torch.Tensor] = []
        for class_name in episode_class_names:
            stage_text_features = self._stage_text_features[class_name].to(device)
            self._require_finite_tensor(
                f"评估类别 '{class_name}' 的语义文本特征",
                stage_text_features,
            )
            if (
                stage_text_features.ndim != 2
                or stage_text_features.shape[1] != feature_dim
            ):
                raise SemanticAlignmentError(
                    f"评估类别 '{class_name}' 的文本/视频特征维度不一致："
                    f"text={tuple(stage_text_features.shape)}, video_dim={feature_dim}"
                )

            sim_flat = cos_sim(
                frames_flat, stage_text_features
            ) / self.semantic_temperature
            self._require_finite_tensor(
                f"评估类别 '{class_name}' 的温度缩放相似度", sim_flat
            )
            similarity_matrix = sim_flat.reshape(target_bs, num_frames, -1)
            dists = 1 - torch.sigmoid(similarity_matrix)
            self._require_finite_tensor(
                f"评估类别 '{class_name}' 的语义距离", dists
            )
            if not self.semantic_transpose:
                dists_4d = dists.transpose(-1, -2).unsqueeze(1)
            else:
                dists_4d = dists.unsqueeze(1)
            cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)
            self._require_finite_tensor(
                f"评估类别 '{class_name}' 的 OTAM 累积距离", cum_dists
            )
            if cum_dists.shape != (target_bs, 1):
                raise SemanticAlignmentError(
                    f"评估类别 '{class_name}' 的 OTAM 输出维度错误："
                    f"shape={tuple(cum_dists.shape)}, expected=({target_bs}, 1)"
                )
            semantic_cum_list.append(cum_dists.squeeze(1))

        semantic_cum = torch.stack(semantic_cum_list, dim=1)
        self._require_finite_tensor("评估语义 OTAM 距离", semantic_cum)
        semantic_probs = F.softmax(-semantic_cum, dim=1)
        self._require_finite_tensor("评估语义对齐概率", semantic_probs)

        # 保持现有概率公式与 TEXT_COFF 的既有解释不变。
        if hasattr(self.args.TRAIN, "TEXT_COFF") and self.args.TRAIN.TEXT_COFF:
            fused_dists = -(
                semantic_probs.pow(self.args.TRAIN.TEXT_COFF)
                * visual_probs.pow(1.0 - self.args.TRAIN.TEXT_COFF)
            )
        else:
            fused_dists = -(semantic_probs.pow(0.5) * visual_probs.pow(0.5))
        fused_logits = -fused_dists
        self._require_finite_tensor("评估语义-视觉融合 logits", fused_logits)

        # 所有检查通过后再原子性替换，异常时不会污染原始帧对齐 logits。
        fused_model_dict = dict(model_dict)
        fused_model_dict["logits"] = fused_logits
        return fused_model_dict

    def forward(self, inputs):
        """
        Few-shot学习的前向传播

        优化说明：整个 forward 只调用一次 get_feats（backbone 前向）。
        - 训练时：从原来的 2 次减少到 1 次，OTAM 路径与语义对齐路径共享同一份原始特征。
        - 评估时：从原来的 3 次减少到 2 次，子类预计算一次特征供语义对齐和融合复用。
        """
        support_images, support_labels, target_images, support_real_class = \
            inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']

        if self.training:
            # ===== 训练模式：单次 backbone 前向，所有路径共享原始特征 =====
            support_features_raw, target_features_raw, _ = self.get_feats(
                support_images, target_images, support_labels)
            support_bs = support_features_raw.shape[0]
            target_bs = target_features_raw.shape[0]

            # ----- 路径1：OTAM 分类（逻辑与父类 CNN_OTAM_SF_CLIP.forward 训练分支完全一致）-----
            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
                feature_classification_in = torch.cat([support_features_raw, target_features_raw], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train) * self.scale
            else:
                class_text_logits = None

            # 根据配置选择使用可学习的 query_embed 或冻结的 text_features
            if self.use_query_embed_train:
                context_support = self.query_embed_train.weight[support_real_class.long()].unsqueeze(1)
            else:
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)

            # target 经 context2 增强（OTAM 路径与语义对齐路径共享此结果，只计算一次）
            target_features_ctx = self.context2(target_features_raw, target_features_raw, target_features_raw)

            # 根据配置决定是否对 context_support 应用 mid_layer
            if self.use_mid_layer_train:
                context_support1 = self.mid_layer(context_support)
            else:
                context_support1 = context_support

            # 并行处理：context_support 同时进入 mid_layer2
            context_support2 = None
            if self.use_mid_layer2:
                context_support2 = self.mid_layer2(context_support)

            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                unique_labels = torch.unique(support_labels)
                support_merged = torch.stack([
                    torch.mean(torch.index_select(support_features_raw, 0, extract_class_indices(support_labels, c)),
                               dim=0)
                    for c in unique_labels])
                ctx_merged = torch.stack([
                    torch.mean(torch.index_select(context_support1, 0, extract_class_indices(support_labels, c)), dim=0)
                    for c in unique_labels])
                # OTAM 用支持集：含文本上下文
                support_features_otam = self.context2(
                    torch.cat([support_merged, ctx_merged], dim=1),
                    torch.cat([support_merged, ctx_merged], dim=1),
                    torch.cat([support_merged, ctx_merged], dim=1)
                )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                # 语义对齐用支持集：纯视觉，不含文本上下文
                support_enhanced_sem = self.context2(support_merged, support_merged, support_merged)
                support_features_for_semantic = torch.stack([
                    support_enhanced_sem[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                    for i in range(support_bs)])
                # 使用 context_support2 强化语义对齐特征（类似 OTAM 使用 context_support1）
                if self.use_mid_layer2_semantic_train and self.use_mid_layer2:
                    ctx_merged2 = torch.stack([
                        torch.mean(torch.index_select(context_support2, 0, extract_class_indices(support_labels, c)),
                                   dim=0)
                        for c in unique_labels])
                    support_enhanced_sem2 = self.context2(
                        torch.cat([support_merged, ctx_merged2], dim=1),
                        torch.cat([support_merged, ctx_merged2], dim=1),
                        torch.cat([support_merged, ctx_merged2], dim=1)
                    )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                    support_features_for_semantic = torch.stack([
                        support_enhanced_sem2[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
            else:
                # OTAM 用支持集：含文本上下文
                support_features_otam = self.context2(
                    torch.cat([support_features_raw, context_support1], dim=1),
                    torch.cat([support_features_raw, context_support1], dim=1),
                    torch.cat([support_features_raw, context_support1], dim=1)
                )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                unique_labels = torch.unique(support_labels)
                support_features_otam = torch.stack([
                    torch.mean(torch.index_select(support_features_otam, 0, extract_class_indices(support_labels, c)),
                               dim=0)
                    for c in unique_labels])
                # 语义对齐用支持集：纯视觉，不含文本上下文
                support_features_for_semantic = self.context2(
                    support_features_raw, support_features_raw, support_features_raw)
                # 使用 context_support2 强化语义对齐特征（类似 OTAM 使用 context_support1）
                if self.use_mid_layer2_semantic_train and self.use_mid_layer2:
                    support_features_for_semantic = self.context2(
                        torch.cat([support_features_raw, context_support2], dim=1),
                        torch.cat([support_features_raw, context_support2], dim=1),
                        torch.cat([support_features_raw, context_support2], dim=1)
                    )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]

            unique_labels = torch.unique(support_labels)
            n_queries = target_features_ctx.shape[0]
            n_support = support_features_otam.shape[0]

            support_flat = rearrange(support_features_otam, 'b s d -> (b s) d')
            target_flat = rearrange(target_features_ctx, 'b s d -> (b s) d')

            frame_sim = cos_sim(target_flat, support_flat)
            frame_dists = 1 - frame_sim
            dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)

            if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                cum_dists = OTAM_cum_dist_v2(dists)
            else:
                cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(
                    rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

            class_dists = torch.stack([
                torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1)
                for c in unique_labels])
            class_dists = rearrange(class_dists, 'c q -> q c')
            model_dict = {'logits': -class_dists, 'class_logits': class_text_logits}

            # ----- 路径2：语义对齐损失（target 特征复用路径1的 target_features_ctx，只计算一次）-----
            if self.semantic_loss_weight > 0:
                try:
                    all_features = torch.cat(
                        [support_features_for_semantic, target_features_ctx], dim=0
                    )
                    all_class_names, _ = self._build_semantic_episode_class_names(
                        inputs, support_bs, target_bs
                    )

                    if not hasattr(self, '_debug_count'):
                        self._debug_count = 0
                    debug_semantic = self._debug_count < 3
                    if debug_semantic:
                        self._debug_count += 1

                    semantic_loss = self._compute_semantic_alignment_loss_multi_class(
                        all_features, all_class_names, debug=debug_semantic
                    )
                    model_dict['semantic_alignment_loss'] = semantic_loss
                    model_dict['semantic_alignment_status'] = 'active'
                except Exception as exc:
                    model_dict = self._handle_semantic_failure(
                        exc,
                        phase='train',
                        model_dict=model_dict,
                        reference_tensor=support_features_for_semantic,
                    )
            else:
                model_dict['semantic_alignment_status'] = 'disabled'

        else:
            # ===== 评估模式 =====
            use_eval_text = hasattr(self.args.TRAIN, "EVAL_TEXT") and self.args.TRAIN.EVAL_TEXT
            use_combine = hasattr(self.args.TRAIN, "COMBINE") and self.args.TRAIN.COMBINE

            support_features_raw, target_features_raw, _ = self.get_feats(
                support_images, target_images, support_labels)
            support_bs = support_features_raw.shape[0]
            target_bs = target_features_raw.shape[0]

            if use_eval_text or use_combine:
                # 非默认评估路径（EVAL_TEXT / COMBINE）：使用父类 forward 计算 logits
                # get_feats 共调用 2 次（此处 1 次 + super 内部 1 次）
                model_dict = super(CNN_SEMANTIC_ALIGNMENT_FEW_SHOT, self).forward(inputs)
                # target context2（供语义对齐使用）
                target_features_ctx = self.context2(
                    target_features_raw, target_features_raw, target_features_raw)
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_merged = torch.stack([
                        torch.mean(
                            torch.index_select(support_features_raw, 0, extract_class_indices(support_labels, c)),
                            dim=0)
                        for c in unique_labels])
                    support_enhanced = self.context2(support_merged, support_merged, support_merged)
                    support_features_for_semantic = torch.stack([
                        support_enhanced[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
                else:
                    support_enhanced = self.context2(
                        support_features_raw, support_features_raw, support_features_raw)
                    unique_labels = torch.unique(support_labels)
                    support_enhanced_cls = torch.stack([
                        torch.mean(torch.index_select(support_enhanced, 0, extract_class_indices(support_labels, c)),
                                   dim=0)
                        for c in unique_labels])
                    support_features_for_semantic = torch.stack([
                        support_enhanced_cls[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
            else:
                # ===== 默认评估路径：内联父类 OTAM 逻辑，context2(target) 只计算一次 =====
                # get_feats 只调用 1 次（上方），不再调用 super().forward()
                # 注意：父类默认评估路径中 context_support 不经过 mid_layer（与训练路径不同）
                feature_classification_in = torch.cat([support_features_raw, target_features_raw], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train) * self.scale

                # 根据配置选择使用 query_embed 或冻结的 text_features
                if self.use_query_embed_eval:
                    context_support = self.query_embed_train.weight[support_real_class.long()].unsqueeze(1)
                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)

                # 根据配置决定是否对 context_support 应用 mid_layer
                if self.use_mid_layer_eval:
                    context_support1 = self.mid_layer(context_support)
                else:
                    context_support1 = context_support

                # 并行处理：context_support 同时进入 mid_layer2
                context_support2 = None
                if self.use_mid_layer2:
                    context_support2 = self.mid_layer2(context_support)

                # target context2：OTAM 路径与语义对齐路径共享，只计算一次（问题4核心优化）
                target_features_ctx = self.context2(
                    target_features_raw, target_features_raw, target_features_raw)

                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_merged = torch.stack([
                        torch.mean(
                            torch.index_select(support_features_raw, 0, extract_class_indices(support_labels, c)),
                            dim=0)
                        for c in unique_labels])
                    ctx_merged = torch.stack([
                        torch.mean(torch.index_select(context_support1, 0, extract_class_indices(support_labels, c)),
                                   dim=0)
                        for c in unique_labels])
                    support_features_otam = self.context2(
                        torch.cat([support_merged, ctx_merged], dim=1),
                        torch.cat([support_merged, ctx_merged], dim=1),
                        torch.cat([support_merged, ctx_merged], dim=1)
                    )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                    # 语义对齐用支持集：纯视觉（不含文本上下文）
                    support_enhanced_sem = self.context2(support_merged, support_merged, support_merged)
                    support_features_for_semantic = torch.stack([
                        support_enhanced_sem[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
                    # 使用 context_support2 强化语义对齐特征（类似 OTAM 使用 context_support1）
                    if self.use_mid_layer2_semantic_eval and self.use_mid_layer2:
                        ctx_merged2 = torch.stack([
                            torch.mean(
                                torch.index_select(context_support2, 0, extract_class_indices(support_labels, c)),
                                dim=0)
                            for c in unique_labels])
                        support_enhanced_sem2 = self.context2(
                            torch.cat([support_merged, ctx_merged2], dim=1),
                            torch.cat([support_merged, ctx_merged2], dim=1),
                            torch.cat([support_merged, ctx_merged2], dim=1)
                        )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                        support_features_for_semantic = torch.stack([
                            support_enhanced_sem2[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                            for i in range(support_bs)])
                else:
                    support_features_otam = self.context2(
                        torch.cat([support_features_raw, context_support1], dim=1),
                        torch.cat([support_features_raw, context_support1], dim=1),
                        torch.cat([support_features_raw, context_support1], dim=1)
                    )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                    unique_labels = torch.unique(support_labels)
                    support_features_otam = torch.stack([
                        torch.mean(
                            torch.index_select(support_features_otam, 0, extract_class_indices(support_labels, c)),
                            dim=0)
                        for c in unique_labels])
                    # 语义对齐用支持集：纯视觉（经过 context2 增强后按类平均，与原始评估保持一致）
                    support_enhanced = self.context2(
                        support_features_raw, support_features_raw, support_features_raw)
                    support_enhanced_cls = torch.stack([
                        torch.mean(torch.index_select(support_enhanced, 0, extract_class_indices(support_labels, c)),
                                   dim=0)
                        for c in unique_labels])
                    support_features_for_semantic = torch.stack([
                        support_enhanced_cls[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
                    # 使用 context_support2 强化语义对齐特征（类似 OTAM 使用 context_support1）
                    if self.use_mid_layer2_semantic_eval and self.use_mid_layer2:
                        support_features_for_semantic = self.context2(
                            torch.cat([support_features_raw, context_support2], dim=1),
                            torch.cat([support_features_raw, context_support2], dim=1),
                            torch.cat([support_features_raw, context_support2], dim=1)
                        )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]

                # OTAM 计算
                unique_labels = torch.unique(support_labels)
                n_queries = target_features_ctx.shape[0]
                n_support = support_features_otam.shape[0]

                support_flat = rearrange(support_features_otam, 'b s d -> (b s) d')
                target_flat = rearrange(target_features_ctx, 'b s d -> (b s) d')
                frame_sim = cos_sim(target_flat, support_flat)
                frame_dists = 1 - frame_sim
                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb=n_queries, sb=n_support)

                if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                    cum_dists = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(
                        rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

                class_dists = torch.stack([
                    torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1)
                    for c in unique_labels])
                class_dists = rearrange(class_dists, 'c q -> q c')
                model_dict = {'logits': -class_dists, 'class_logits': class_text_logits}

            # ----- 路径2：语义损失和评估概率融合使用同一个异常边界 -----
            semantic_combine = bool(
                getattr(self.args.TRAIN, 'SEMANTIC_COMBINE', False)
            )
            if semantic_combine:
                visual_model_dict = dict(model_dict)
                try:
                    all_features = torch.cat(
                        [support_features_for_semantic, target_features_ctx], dim=0
                    )
                    all_class_names, _ = self._build_semantic_episode_class_names(
                        inputs, support_bs, target_bs
                    )
                    with torch.no_grad():
                        semantic_alignment_loss = self._compute_semantic_alignment_loss_multi_class(
                            all_features, all_class_names, debug=False
                        )
                    semantic_model_dict = dict(model_dict)
                    semantic_model_dict['semantic_alignment_loss'] = semantic_alignment_loss
                    model_dict = self._fuse_semantic_and_visual_probs_eval(
                        inputs,
                        semantic_model_dict,
                        cached_target_features=target_features_raw,
                    )
                    model_dict['semantic_alignment_status'] = 'active'
                except Exception as exc:
                    model_dict = self._handle_semantic_failure(
                        exc,
                        phase='eval',
                        model_dict=visual_model_dict,
                    )
            else:
                model_dict['semantic_alignment_status'] = 'disabled'

        return model_dict

    # def loss(self, task_dict, model_dict):
    #     """
    #     计算总损失：原始few-shot损失 + 语义对齐损失
    #
    #     损失构成说明：
    #     1. base_loss: Few-shot分类损失 (帧对齐，support-query对齐)
    #     2. semantic_alignment_loss: 语义对齐损失 (帧-语义阶段对齐)
    #
    #     Args:
    #         task_dict: 包含标签的字典
    #         model_dict: 包含模型输出的字典
    #
    #     Returns:
    #         dict: 包含各个损失项的字典
    #     """
    #     # 基础few-shot损失（父类的帧对齐损失）
    #     base_loss = F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
    #
    #     # 语义对齐损失
    #     semantic_loss = model_dict.get("semantic_alignment_loss", torch.tensor(0.0, device=base_loss.device))
    #
    #     # 总损失
    #     total_loss = base_loss + self.semantic_loss_weight * semantic_loss
    #
    #     # 应用batch size缩放（如果配置了）
    #     if hasattr(self.args.TRAIN, 'BATCH_SIZE'):
    #         total_loss = total_loss / self.args.TRAIN.BATCH_SIZE
    #
    #     # 返回所有损失组件
    #     return {
    #         'total_loss': total_loss,
    #         'frame_alignment_loss': base_loss,  # few-shot帧对齐损失
    #         'semantic_loss': semantic_loss,
    #         'weighted_semantic_loss': self.semantic_loss_weight * semantic_loss
    #     }
