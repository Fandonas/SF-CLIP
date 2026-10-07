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


class CNN_SEMANTIC_ALIGNMENT_SUPERVISED(CNN_OTAM_SF_CLIP):
    """
    全监督学习的语义对齐模型
    
    核心改进：
    1. 损失函数构成：第一个损失（视频-文本相似度） + 语义对齐损失（OTAM算法）
    2. 使用CLIP进行视频帧和文本的对齐
    3. 改为全监督学习，不再使用few-shot
    4. 帧级别语义对齐：每一帧都能与具体的语义阶段进行对齐，不进行帧分组
    
    语义对齐机制：
    - 将动作类别名称解析为多个语义阶段
    - 使用CLIP对每个语义阶段进行文本编码
    - 直接使用每一帧特征与语义阶段进行OTAM对齐
    - 找到帧与语义阶段的最优对齐路径
    """
    
    def __init__(self, args):
        super().__init__(args)
        
        # 获取完整的CLIP模型用于文本编码
        from models.base.few_shot import load
        self.full_clip_model, _ = load(args.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=args, jit=False)
        
        # 语义阶段解析
        self.semantic_stages = self._parse_semantic_stages()
        
        # 为每个阶段创建文本特征
        self.stage_text_features = self._create_stage_text_features()
        
        # 语义对齐损失权重
        self.semantic_loss_weight = getattr(args.TRAIN, 'SEMANTIC_LOSS_WEIGHT', 0.5)
        
    def _parse_semantic_stages(self) -> Dict[str, List[str]]:#正确
        """解析类别名称的语义阶段"""
        semantic_stages = {}
        
        for class_name in self.class_real_train:
            # 使用逗号和"then"分割语义阶段
            stages = re.split(r'[,，]\s*|then\s+', class_name.lower())
            stages = [stage.strip() for stage in stages if stage.strip()]
            """
            这行代码的作用是：
            去除每个字符串的首尾空白。
            过滤掉空字符串（包括只包含空白的字符串）。
            返回一个干净的新列表
            """
            if len(stages) > 1:
                semantic_stages[class_name] = stages
            else:
                # 如果没有明确的分割，保持原始名称
                semantic_stages[class_name] = [class_name]
                
        return semantic_stages
    
    def _create_stage_text_features(self) -> Dict[str, torch.Tensor]:#正确吧
        """为每个类别的语义阶段创建文本特征"""
        stage_text_features = {}
        
        # 导入tokenize函数
        from models.base.few_shot import tokenize
        
        # 使用torch.no_grad()避免梯度计算问题
        with torch.no_grad():
            for class_name, stages in self.semantic_stages.items():
                stage_features = []
                for stage in stages:
                    # 使用CLIP编码每个阶段的文本
                    stage_text = f"a video of {stage}" #改为视频
                    # 使用完整的CLIP模型进行文本编码
                    text_tokens = tokenize([stage_text]).cuda()
                    stage_feature = self.full_clip_model.encode_text(text_tokens)
                    stage_features.append(stage_feature)
                
                # 堆叠所有阶段特征 [num_stages, feature_dim]
                stage_text_features[class_name] = torch.cat(stage_features, dim=0)
            
        return stage_text_features
    
    def _compute_semantic_alignment_loss(self, video_features: torch.Tensor, 
                                       class_labels: torch.Tensor, debug=False) -> torch.Tensor:
        """
        计算语义对齐损失，使用OTAM算法进行帧级别的语义阶段对齐
        
        核心改进：
        1. 不对视频帧进行分组，直接使用每一帧
        2. 每一帧都能与具体的语义阶段进行对齐
        3. 使用OTAM算法找到帧与语义阶段的最优对齐路径
        
        Args:
            video_features: [batch_size, num_frames, feature_dim] 视频特征
            class_labels: [batch_size] 类别标签
            
        Returns:
            semantic_loss: 语义对齐损失
        """
        try:
            batch_size, num_frames, feature_dim = video_features.shape
            
            # 使用列表收集所有损失，避免in-place操作
            semantic_losses = []
            
            for i in range(batch_size):
                class_idx = class_labels[i].item()
                # 确保class_idx在有效范围内
                if class_idx < len(self.class_real_train):
                    class_name = self.class_real_train[class_idx]
                    
                    # 获取该类别的语义阶段文本特征
                    if class_name in self.stage_text_features:
                        stage_text_features = self.stage_text_features[class_name]  # [num_stages, feature_dim]
                        num_stages = stage_text_features.shape[0]
                        
                        # 直接使用每一帧，不进行分组
                        frame_features = video_features[i]  # [num_frames, feature_dim]
                        
                        # 计算每一帧与每个语义阶段的相似度矩阵
                        # frame_features: [num_frames, feature_dim]
                        # stage_text_features: [num_stages, feature_dim]
                        print(frame_features.shape)
                        print(stage_text_features.shape)#改

                        similarity_matrix = cos_sim(frame_features, stage_text_features)  # [num_frames, num_stages]
                        
                        # 调试信息：检查相似度矩阵的范围
                        if debug and i == 0:  # 只在第一个样本打印
                            print(f"Debug - Similarity matrix range: [{similarity_matrix.min().item():.4f}, {similarity_matrix.max().item():.4f}]")
                        
                        # 使用OTAM算法找到帧与语义阶段的最优对齐路径
                        dists = 1 - similarity_matrix  # 转换为距离（越小越好）
                        
                        # 调试信息：检查距离矩阵的范围
                        if debug and i == 0:  # 只在第一个样本打印
                            print(f"Debug - Distance matrix range: [{dists.min().item():.4f}, {dists.max().item():.4f}]")
                        
                        # 将2D距离矩阵转换为4D张量：[1, 1, num_frames, num_stages]
                        dists_4d = dists.unsqueeze(0).unsqueeze(0)  # [1, 1, num_frames, num_stages]
                        cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)  # [1, 1]
                        
                        # 调试信息：检查OTAM输出
                        if debug and i == 0:  # 只在第一个样本打印
                            print(f"Debug - OTAM output: {cum_dists[0, 0].item():.4f}")
                        
                        # 语义对齐损失：最小化帧与语义阶段的最优对齐路径总距离
                        # OTAM返回负值表示更好的对齐，我们需要取绝对值得到正的距离
                        semantic_loss = torch.abs(cum_dists[0, 0])  # 取OTAM距离的绝对值
                        semantic_losses.append(semantic_loss)
            
            if len(semantic_losses) > 0:
                # 使用torch.stack避免in-place操作
                all_losses = torch.stack(semantic_losses)
                return all_losses.mean()
            else:
                return torch.tensor(0.0, device=video_features.device, requires_grad=True)
                
        except Exception as e:
            print(f"Warning: Semantic alignment loss computation failed: {e}")
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)
    
    def forward(self, support_images, target_images, support_labels, target_labels=None):
        """
        全监督学习的前向传播
        
        Args:
            support_images: [batch_size, num_frames, C, H, W] 支持集图像（在全监督中作为训练样本）
            target_images: [batch_size, num_frames, C, H, W] 目标图像（在全监督中作为验证样本）
            support_labels: [batch_size] 支持集标签
            target_labels: [batch_size] 目标标签（可选）
        """
        # 重塑视频数据以适应backbone的输入格式
        batch_size, num_frames, C, H, W = support_images.shape
        
        # 将视频帧重塑为 [batch_size * num_frames, C, H, W]
        support_images_reshaped = support_images.view(batch_size * num_frames, C, H, W)
        target_images_reshaped = target_images.view(batch_size * num_frames, C, H, W)
        
        # 获取视频特征
        support_features, target_features, _ = self.get_feats(
            support_images_reshaped, target_images_reshaped, support_labels
        )
        
        # 重塑特征回到 [batch_size, num_frames, feature_dim]
        support_features = support_features.view(batch_size, num_frames, -1)
        target_features = target_features.view(batch_size, num_frames, -1)
        
        # 第一个损失：视频-文本相似度损失（原始模型的第一个损失）
        # 计算支持集和目标集的视频特征与文本特征的相似度
        all_features = torch.cat([support_features, target_features], dim=0)
        all_labels = torch.cat([support_labels, target_labels if target_labels is not None else support_labels], dim=0)
        
        # 使用分类层处理特征（与原始模型保持一致）
        feature_classification = self.classification_layer(all_features).mean(1)  # [batch_size, feature_dim]
        
        # 计算与文本特征的相似度
        # 使用温度参数来控制相似度的缩放
        temperature = 0.1  # 使用更小的温度来放大差异
        text_similarity = cos_sim(feature_classification, self.text_features_train) / temperature
        text_logits = text_similarity  # 使用温度缩放后的相似度作为logits
        
        # 调试信息：检查文本相似度的范围
        if hasattr(self, '_debug_text_count') and self._debug_text_count < 3:
            if not hasattr(self, '_debug_text_count'):
                self._debug_text_count = 0
            self._debug_text_count += 1
            print(f"Debug - Text similarity range: [{text_similarity.min().item():.4f}, {text_similarity.max().item():.4f}]")
            print(f"Debug - Scale value: {self.scale.item():.4f}")
            print(f"Debug - Feature classification norm: {torch.norm(feature_classification, dim=1).mean().item():.4f}")
            print(f"Debug - Text features norm: {torch.norm(self.text_features_train, dim=1).mean().item():.4f}")
            
            # 检查当前batch的标签和对应的相似度
            print(f"Debug - Current batch labels: {all_labels[:5].tolist()}")
            print(f"Debug - Text logits shape: {text_logits.shape}")
            print(f"Debug - Text logits sample (first 5 samples, first 5 classes):")
            print(text_logits[:5, :5].detach().cpu().numpy())
            
            # 检查正确类别的相似度
            for i in range(min(3, len(all_labels))):
                label = all_labels[i].item()
                correct_similarity = text_logits[i, label].item()
                print(f"Debug - Sample {i}, Label {label}, Correct similarity: {correct_similarity:.4f}")
                
            # 检查交叉熵损失的输入
            print(f"Debug - Cross entropy input range: [{text_logits.min().item():.4f}, {text_logits.max().item():.4f}]")
        
        # 第二个损失：语义对齐损失（使用OTAM算法）
        # 在训练初期启用调试
        debug_semantic = hasattr(self, '_debug_count') and self._debug_count < 3
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if debug_semantic:
            self._debug_count += 1
            
        semantic_alignment_loss = self._compute_semantic_alignment_loss(all_features, all_labels, debug=debug_semantic)
        
        # 计算分类logits（用于评估）
        # 简化版本：直接使用text_similarity作为logits
        class_dists = -text_similarity  # 转换为距离（越小越好）
        
        return {
            'logits': class_dists,
            'text_logits': text_logits,
            'semantic_alignment_loss': semantic_alignment_loss,
            'video_features': all_features
        }
    
    def loss(self, task_dict, model_dict):
        """
        计算总损失：第一个损失（视频-文本相似度） + 语义对齐损失
        
        Args:
            task_dict: 包含标签的字典
            model_dict: 包含模型输出的字典
            
        Returns:
            total_loss: 总损失
        """
        # 第一个损失：视频-文本相似度损失
        # 确保标签维度匹配
        target_labels = task_dict["target_labels"].long()
        text_logits = model_dict["text_logits"]
        
        # 如果批次大小不匹配，调整text_logits
        if text_logits.shape[0] != target_labels.shape[0]:
            # 取前N个样本，其中N是target_labels的批次大小
            text_logits = text_logits[:target_labels.shape[0]]
        
        # 确保标签值在有效范围内
        num_classes = text_logits.shape[1]
        target_labels = torch.clamp(target_labels, 0, num_classes - 1)
        
        text_similarity_loss = F.cross_entropy(text_logits, target_labels)
        
        # 第二个损失：语义对齐损失
        semantic_loss = model_dict["semantic_alignment_loss"]
        
        # 总损失
        total_loss = text_similarity_loss + self.semantic_loss_weight * semantic_loss
        
        # 返回总损失和各个损失组件
        return {
            'total_loss': total_loss,
            'text_similarity_loss': text_similarity_loss,
            'semantic_loss': semantic_loss,
            'weighted_semantic_loss': self.semantic_loss_weight * semantic_loss
        }
