import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
import sys
import os

# 
from models.base.few_shot import CNN_OTAM_SF_CLIP, cos_sim, OTAM_cum_dist_v2
from models.base.few_shot import HEAD_REGISTRY
from einops import rearrange
from models.base.few_shot import extract_class_indices


@HEAD_REGISTRY.register()
class CNN_SEMANTIC_ALIGNMENT_FEW_SHOT(CNN_OTAM_SF_CLIP):
    """
    

    
    1. OTAMsupportquery
    2. 
    3. few-shot + OTAM
    4. few-shotsupporttarget

    
    - 
    - CLIP
    - OTAM
    - 

    
    """

    def __init__(self, cfg):
        super(CNN_SEMANTIC_ALIGNMENT_FEW_SHOT, self).__init__(cfg)

        # 
        self.semantic_loss_weight = getattr(cfg.TRAIN, 'SEMANTIC_LOSS_WEIGHT', 0.5)

        # 
        # temperaturesharpsmooth
        self.semantic_temperature = getattr(cfg.TRAIN, 'SEMANTIC_TEMPERATURE', 0.1)
        # True(=)False(=)
        self.semantic_transpose = getattr(cfg.TRAIN, 'SEMANTIC_TRANSPOSE', False)

        # Query Embedding 
        self.use_query_embed_train = getattr(cfg.TRAIN, 'USE_QUERY_EMBED_TRAIN', True)
        self.use_query_embed_eval = getattr(cfg.TRAIN, 'USE_QUERY_EMBED_EVAL', False)

        # Mid Layer 
        self.use_mid_layer_train = getattr(cfg.TRAIN, 'USE_MID_LAYER_TRAIN', True)
        self.use_mid_layer_eval = getattr(cfg.TRAIN, 'USE_MID_LAYER_EVAL', False)
        self.use_mid_layer2 = getattr(cfg.TRAIN, 'USE_MID_LAYER2', False)
        self.use_mid_layer2_semantic_train = getattr(cfg.TRAIN, 'USE_MID_LAYER2_SEMANTIC_TRAIN', False)
        self.use_mid_layer2_semantic_eval = getattr(cfg.TRAIN, 'USE_MID_LAYER2_SEMANTIC_EVAL', False)
        self.use_mid_layer_relu = getattr(cfg.TRAIN, 'USE_MID_LAYER_RELU', False)

        # CUDA
        self._full_clip_model = None
        self._semantic_stages = None
        self._stage_text_features = None
        self._text_features_initialized = False

        #  query_embed CLIP 
        #  [num_train_classes, mid_dim]64512=32K
        #  text_features_testnovel
        num_train_classes = self.text_features_train.shape[0]
        if self.use_query_embed_train:
            #  EmbeddingDDP 
            # use_query_embed_eval=True  use_query_embed_train 
            self.query_embed_train = nn.Embedding(num_train_classes, self.mid_dim)
            self.query_embed_train.weight = nn.Parameter(
                self.text_features_train.detach().clone()
            )
        # use_query_embed_train=False
        # -  text_features_trainfrozen bufferDDP 
        # -  text_features_testquery_embed_train 
        # -  use_query_embed_eval=True  use_query_embed_train=False
        #    eval forward  AttributeError

        #  nn.Sequential()
        #  False mid_layer = nn.Sequential()0DDP 
        if self.use_mid_layer_train or self.use_mid_layer_eval:
            if self.use_mid_layer_relu:
                self.mid_layer = nn.Sequential(
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.ReLU()
                )
            else:
                self.mid_layer = nn.Linear(self.mid_dim, self.mid_dim)

        #  mid_layer2 mid_layer  context_support
        if self.use_mid_layer2:
            if self.use_mid_layer_relu:
                self.mid_layer2 = nn.Sequential(
                    nn.Linear(self.mid_dim, self.mid_dim),
                    nn.ReLU()
                )
            else:
                self.mid_layer2 = nn.Linear(self.mid_dim, self.mid_dim)

    def _get_class_name(self, label_idx: int) -> str:
        """
        label_idx

        
        - ID0
        - (self.training=True): label_idx  [0, 30]  class_real_train[label_idx]
        - (self.training=False): label_idx  [0, 9]  class_real_test[label_idx]

        Args:
            label_idx: real_support_labelsreal_target_labels

        Returns:
            
        """
        label_idx = int(label_idx)

        # training
        if self.training:
            # 
            if 0 <= label_idx < len(self.class_real_train):
                return self.class_real_train[label_idx]
            else:
                # 
                print(
                    f"Warning: label_idx {label_idx} out of range for train classes (0-{len(self.class_real_train) - 1})")
                return self.class_real_train[0] if len(self.class_real_train) > 0 else ""
        else:
            # /
            if 0 <= label_idx < len(self.class_real_test):
                return self.class_real_test[label_idx]
            else:
                # 
                print(
                    f"Warning: label_idx {label_idx} out of range for test classes (0-{len(self.class_real_test) - 1})")
                return self.class_real_test[0] if len(self.class_real_test) > 0 else ""

    def _ensure_text_features_initialized(self):
        """"""
        if self._text_features_initialized:
            return

        try:
            # CLIP
            from models.base.few_shot import load

            # 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._full_clip_model, _ = load(self.args.VIDEO.HEAD.BACKBONE_NAME, device=device, cfg=self.args, jit=False)

            # 
            self._semantic_stages = self._parse_semantic_stages()

            # 
            self._stage_text_features = self._create_stage_text_features()  # 

            #  _stage_text_features_full_clip_model 
            del self._full_clip_model
            self._full_clip_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._text_features_initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize text features: {e}")
            import traceback
            traceback.print_exc()
            # 
            self._semantic_stages = {}
            self._stage_text_features = {}
            self._text_features_initialized = True  # 

    def _parse_semantic_stages(self) -> Dict[str, List[str]]:  # "," ", " ", then "
        """"""
        semantic_stages = {}

        # few-shot
        all_classes = self.class_real_train + self.class_real_test

        for class_name in all_classes:
            # "then"
            stages = re.split(r'[,]\s*|then\s+', class_name.lower())
            stages = [stage.strip() for stage in stages if stage.strip()]

            if len(stages) > 1:
                semantic_stages[class_name] = stages
            else:
                # 
                semantic_stages[class_name] = [class_name]

        return semantic_stages

    def _create_stage_text_features(self) -> Dict[str, torch.Tensor]:  # cat
        """"""
        stage_text_features = {}

        # tokenize
        from models.base.few_shot import tokenize

        # 
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # torch.no_grad()
        with torch.no_grad():
            for class_name, stages in self._semantic_stages.items():
                stage_features = []
                for stage in stages:
                    try:
                        # CLIP
                        stage_text = f"a video of {stage}"
                        # CLIP
                        text_tokens = tokenize([stage_text]).to(device)
                        stage_feature = self._full_clip_model.encode_text(text_tokens)
                        stage_features.append(stage_feature)
                    except Exception as e:
                        print(f"Warning: Failed to encode stage '{stage}' for class '{class_name}': {e}")
                        continue

                if len(stage_features) > 0:
                    #  [num_stages, feature_dim]
                    stage_text_features[class_name] = torch.cat(stage_features, dim=0)
                else:
                    # 
                    print(f"Warning: No valid features for class '{class_name}', skipping")

        return stage_text_features

    def _compute_semantic_alignment_loss(self, video_features: torch.Tensor,
                                         class_names: List[str], debug=False) -> torch.Tensor:
        """
        OTAM

        
        1. 
        2. 
        3. OTAM

        Args:
            video_features: [batch_size, num_frames, feature_dim] 
            class_names: [batch_size] 

        Returns:
            semantic_loss: 
        """
        try:
            # 
            self._ensure_text_features_initialized()

            if not self._stage_text_features or len(self._stage_text_features) == 0:
                # 0
                return torch.tensor(0.0, device=video_features.device, requires_grad=True)

            batch_size, num_frames, feature_dim = video_features.shape

            # in-place
            semantic_losses = []

            for i in range(batch_size):
                class_name = class_names[i]

                # 
                if class_name in self._stage_text_features:
                    stage_text_features = self._stage_text_features[class_name]  # [num_stages, feature_dim]

                    # stage_text_features
                    stage_text_features = stage_text_features.to(video_features.device)

                    # 
                    frame_features = video_features[i]  # [num_frames, feature_dim]

                    # 
                    # temperature
                    similarity_matrix = cos_sim(frame_features,
                                                stage_text_features) / self.semantic_temperature  # [num_frames, num_stages]

                    # 
                    if debug and i == 0:
                        print(
                            f"Debug - Similarity matrix range (after temperature): [{similarity_matrix.min().item():.4f}, {similarity_matrix.max().item():.4f}]")
                        print(f"Debug - Semantic temperature: {self.semantic_temperature}")

                    # OTAM
                    dists = 1 - torch.sigmoid(similarity_matrix)  # sigmoid

                    # SEMANTIC_TRANSPOSE=False: [1,1,num_stages,num_frames]=
                    # SEMANTIC_TRANSPOSE=True:  [1,1,num_frames,num_stages]=
                    if not self.semantic_transpose:
                        dists_4d = dists.T.unsqueeze(0).unsqueeze(0)  # [1, 1, num_stages, num_frames]
                    else:
                        dists_4d = dists.unsqueeze(0).unsqueeze(0)  # [1, 1, num_frames, num_stages]
                    cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)  # [1, 1]

                    # OTAM
                    if debug and i == 0:
                        print(f"Debug - OTAM output: {cum_dists[0, 0].item():.4f}")

                    # 
                    semantic_loss = torch.abs(cum_dists[0, 0])  # OTAM
                    semantic_losses.append(semantic_loss)

            if len(semantic_losses) > 0:
                # torch.stackin-place
                all_losses = torch.stack(semantic_losses)
                return all_losses.mean()
            else:
                return torch.tensor(0.0, device=video_features.device, requires_grad=True)

        except Exception as e:
            print(f"Warning: Semantic alignment loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)

    def _compute_semantic_alignment_loss_multi_class(self,
                                                     video_features: torch.Tensor,
                                                     class_names: List[str],
                                                     debug: bool = False) -> torch.Tensor:
        """
        
        -  episode  OTAM 
        -  cum_dists[class]
        -  -cum_dists  logits softmax / CrossEntropy 
        -  temperature  sigmoid cos_sim / T  sigmoid  1 - sigmoid

         batch_size  num_classes 
        OTAM  batch_sizenum_classes  num_classes 
        

        Args:
            video_features: [batch_size, num_frames, feature_dim]support  target 
            class_names:    [batch_size] video_features 
        """
        try:
            # 
            self._ensure_text_features_initialized()

            device = video_features.device

            if not self._stage_text_features or len(self._stage_text_features) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            batch_size, num_frames, feature_dim = video_features.shape

            #  episode 
            episode_classes: List[str] = []
            for name in class_names:
                if (name in self._stage_text_features) and (name not in episode_classes):
                    episode_classes.append(name)

            if len(episode_classes) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            #  episode 
            episode_stage_feats = {
                cname: self._stage_text_features[cname].to(device)
                for cname in episode_classes
            }  #  episode 

            # gt_name  episode_stage_feats 
            valid_indices = [i for i in range(batch_size) if class_names[i] in episode_stage_feats]
            if len(valid_indices) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            #  [batch_size * num_frames, feature_dim]
            # 
            frames_flat = video_features.reshape(batch_size * num_frames, feature_dim)

            # ===== num_classes  batchclasses =====
            #  c batch_size  OTAM 
            all_cum_dists_list: List[torch.Tensor] = []

            for cls_idx, cname in enumerate(episode_classes):
                stage_text_features = episode_stage_feats[cname]  # [num_stages, feature_dim]
                num_stages = stage_text_features.shape[0]

                # 
                # cos_sim: [batch_size*num_frames, D]  [num_stages, D]  [batch_size*num_frames, num_stages]
                sim_flat = cos_sim(frames_flat, stage_text_features) / self.semantic_temperature
                #  [batch_size, num_frames, num_stages]
                #  i  [num_frames, num_stages]
                similarity_matrix = sim_flat.reshape(batch_size, num_frames, num_stages)

                if debug and cls_idx == 0:
                    #  i==0, cls_idx==0 
                    sm0 = similarity_matrix[0]
                    print(
                        f"Debug(multi) - Similarity matrix range (after temperature): "
                        f"[{sm0.min().item():.4f}, {sm0.max().item():.4f}]"
                    )
                    print(f"Debug(multi) - Semantic temperature: {self.semantic_temperature}")

                # sigmoid  -> 
                dists = 1 - torch.sigmoid(similarity_matrix)  # [batch_size, num_frames, num_stages]

                # SEMANTIC_TRANSPOSE=False: [B,1,num_stages,num_frames]=
                # SEMANTIC_TRANSPOSE=True:  [B,1,num_frames,num_stages]=
                if not self.semantic_transpose:
                    dists_4d = dists.transpose(-1, -2).unsqueeze(1)  # [batch_size, 1, num_stages, num_frames]
                else:
                    dists_4d = dists.unsqueeze(1)  # [batch_size, 1, num_frames, num_stages]
                cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)  # [batch_size, 1]

                if debug and cls_idx == 0:
                    print(f"Debug(multi) - OTAM output (first video, first class): {cum_dists[0, 0].item():.4f}")

                all_cum_dists_list.append(cum_dists.squeeze(1))  # [batch_size]

            #  [batch_size, num_episode_classes]
            # all_cum_dists[i, c]  i  c  OTAM 
            all_cum_dists = torch.stack(all_cum_dists_list, dim=1)  # [batch_size, C]
            logits_all = -all_cum_dists  # [batch_size, C] -> logit 

            # 
            gt_indices = torch.tensor(
                [episode_classes.index(class_names[i]) for i in valid_indices],
                device=device, dtype=torch.long
            )

            #  CrossEntropyreduction='mean'
            loss = F.cross_entropy(logits_all[valid_indices], gt_indices)
            return loss

        except Exception as e:
            print(f"Warning: Multi-class semantic alignment loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=video_features.device, requires_grad=True)

    def _fuse_semantic_and_visual_probs_eval(self, inputs, model_dict, cached_target_features=None):
        """
        
        " + "
         +  COMBINE 

            p_fused  p_semantic^ * p_visual^(1-)
            logits = -p_fused

        
        - p_visual  model_dict["logits"]  softmaxfew-shot OTAM 
        - p_semantic " vs episode " OTAM 
           -cum_dists  softmax

         episode  query/target 
        """
        #  SEMANTIC_COMBINE 
        if self.training:
            return model_dict
        if not hasattr(self.args.TRAIN, "SEMANTIC_COMBINE") or not self.args.TRAIN.SEMANTIC_COMBINE:
            return model_dict
        if "logits" not in model_dict:
            return model_dict

        try:
            """
                    Few-shot

                    Args:
                        inputs: 
                            - support_set: [support_size, num_frames, C, H, W]
                            - support_labels: [support_size] few-shot
                            - target_set: [target_size, num_frames, C, H, W]
                            - real_support_labels: [support_size] 
            """
            support_images = inputs["support_set"]
            support_labels = inputs["support_labels"]
            target_images = inputs["target_set"]
            support_real_class = inputs["real_support_labels"]

            #  get_featsbackbone 
            if cached_target_features is not None:
                target_features = cached_target_features
            else:
                _, target_features, _ = self.get_feats(
                    support_images, target_images, support_real_class
                )

            device = target_features.device
            target_bs = target_features.shape[0]
            if target_bs == 0:
                return model_dict

            #  support_labels / support_real_class  episode  OTAM 
            unique_labels = torch.unique(support_labels)
            episode_class_names: List[str] = []
            for c in unique_labels:
                #  episodic label  support  id
                idx = extract_class_indices(support_labels, c)[0]
                label_idx = int(support_real_class[idx].item())
                # training
                cname = self._get_class_name(label_idx)
                episode_class_names.append(cname)

            num_classes = len(episode_class_names)
            if num_classes == 0:
                return model_dict

            #  few-shot OTAM logits  softmax
            visual_logits = model_dict["logits"]
            if visual_logits.shape[0] != target_bs or visual_logits.shape[1] != num_classes:
                # 
                return model_dict
            visual_probs = F.softmax(visual_logits, dim=1)  # [target_bs, num_classes]

            #  target  OTAM 
            self._ensure_text_features_initialized()
            if not self._stage_text_features or len(self._stage_text_features) == 0:
                return model_dict

            episode_stage_feats = {}
            for cname in episode_class_names:
                if cname in self._stage_text_features:
                    episode_stage_feats[cname] = self._stage_text_features[cname].to(device)
            if len(episode_stage_feats) != num_classes:
                # 
                return model_dict

            # =====  _compute_semantic_alignment_loss_multi_class  Problem3 =====
            # target_bs  num_classes = 50  OTAM10  5
            # num_classes = 5  OTAM target_bs 
            num_frames = target_features.shape[1]
            feature_dim = target_features.shape[2]
            # [target_bs, num_frames, D]  [target_bs*num_frames, D]
            frames_flat = target_features.reshape(target_bs * num_frames, feature_dim)

            semantic_cum_list: List[torch.Tensor] = []
            for cj, cname in enumerate(episode_class_names):
                stage_text_features = episode_stage_feats[cname]  # [num_stages, D]
                # [target_bs*num_frames, D]  [num_stages, D]  [target_bs*num_frames, num_stages]
                sim_flat = cos_sim(frames_flat, stage_text_features) / self.semantic_temperature
                #  [target_bs, num_frames, num_stages] [num_frames, num_stages] 
                similarity_matrix = sim_flat.reshape(target_bs, num_frames, -1)
                dists = 1 - torch.sigmoid(similarity_matrix)  # [target_bs, num_frames, num_stages]
                # SEMANTIC_TRANSPOSE=False: [target_bs,1,num_stages,num_frames]=
                # SEMANTIC_TRANSPOSE=True:  [target_bs,1,num_frames,num_stages]=
                if not self.semantic_transpose:
                    dists_4d = dists.transpose(-1, -2).unsqueeze(1)  # [target_bs, 1, num_stages, num_frames]
                else:
                    dists_4d = dists.unsqueeze(1)  # [target_bs, 1, num_frames, num_stages]
                cum_dists = OTAM_cum_dist_v2(dists_4d, lbda=0.5)  # [target_bs, 1]
                semantic_cum_list.append(cum_dists.squeeze(1))  # [target_bs]

            semantic_cum = torch.stack(semantic_cum_list, dim=1)  # [target_bs, num_classes]

            # few_shot.py:2894
            # softmax(similarity)softmax(-distance) softmax
            semantic_probs = F.softmax(-semantic_cum, dim=1)  # [target_bs, num_classes]

            #  few_shot.py:2952-2954 + 3016 
            #    1- 
            if hasattr(self.args.TRAIN, "TEXT_COFF") and self.args.TRAIN.TEXT_COFF:
                # 
                fused_dists = -(semantic_probs.pow(self.args.TRAIN.TEXT_COFF) * visual_probs.pow(
                    1.0 - self.args.TRAIN.TEXT_COFF))
            else:
                fused_dists = -(semantic_probs.pow(0.5) * visual_probs.pow(0.5))

            #  few_shot.py:3016  return {'logits': -class_dists}
            #  logits = semantic_probs^ * visual_probs^(1-)
            model_dict["logits"] = -fused_dists
            return model_dict

        except Exception as e:
            print(f"Warning: semantic-visual fusion in eval failed: {e}")
            import traceback
            traceback.print_exc()
            return model_dict

    def forward(self, inputs):
        """
        Few-shot

         forward  get_featsbackbone 
        -  2  1 OTAM 
        -  3  2 
        """
        support_images, support_labels, target_images, support_real_class = \
            inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels']

        if self.training:
            # =====  backbone  =====
            support_features_raw, target_features_raw, _ = self.get_feats(
                support_images, target_images, support_labels)
            support_bs = support_features_raw.shape[0]
            target_bs = target_features_raw.shape[0]

            # ----- 1OTAM  CNN_OTAM_SF_CLIP.forward -----
            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
                feature_classification_in = torch.cat([support_features_raw, target_features_raw], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train) * self.scale
            else:
                class_text_logits = None

            #  query_embed  text_features
            if self.use_query_embed_train:
                context_support = self.query_embed_train.weight[support_real_class.long()].unsqueeze(1)
            else:
                context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)

            # target  context2 OTAM 
            target_features_ctx = self.context2(target_features_raw, target_features_raw, target_features_raw)

            #  context_support  mid_layer
            if self.use_mid_layer_train:
                context_support1 = self.mid_layer(context_support)
            else:
                context_support1 = context_support

            # context_support  mid_layer2
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
                # OTAM 
                support_features_otam = self.context2(
                    torch.cat([support_merged, ctx_merged], dim=1),
                    torch.cat([support_merged, ctx_merged], dim=1),
                    torch.cat([support_merged, ctx_merged], dim=1)
                )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]
                # 
                support_enhanced_sem = self.context2(support_merged, support_merged, support_merged)
                support_features_for_semantic = torch.stack([
                    support_enhanced_sem[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                    for i in range(support_bs)])
                #  context_support2  OTAM  context_support1
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
                # OTAM 
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
                # 
                support_features_for_semantic = self.context2(
                    support_features_raw, support_features_raw, support_features_raw)
                #  context_support2  OTAM  context_support1
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

            # ----- 2target 1 target_features_ctx-----
            try:
                all_features = torch.cat([support_features_for_semantic, target_features_ctx], dim=0)

                all_class_names = []
                for i in range(support_bs):
                    all_class_names.append(self._get_class_name(int(support_real_class[i].item())))
                if 'real_target_labels' in inputs:
                    target_real_labels = inputs['real_target_labels']
                    for i in range(min(target_bs, len(target_real_labels))):
                        all_class_names.append(self._get_class_name(int(target_real_labels[i].item())))
                else:
                    for i in range(min(target_bs, support_bs)):
                        all_class_names.append(self._get_class_name(int(support_real_class[i].item())))
                if target_bs > len(all_class_names) - support_bs:
                    last_class = all_class_names[-1] if all_class_names else (
                        self.class_real_train[0] if self.class_real_train else "")
                    needed = target_bs - (len(all_class_names) - support_bs)
                    for _ in range(needed):
                        all_class_names.append(last_class)

                if not hasattr(self, '_debug_count'):
                    self._debug_count = 0
                debug_semantic = self._debug_count < 3
                if debug_semantic:
                    self._debug_count += 1

                model_dict['semantic_alignment_loss'] = self._compute_semantic_alignment_loss_multi_class(
                    all_features, all_class_names, debug=debug_semantic)
            except Exception as e:
                print(f"Warning: Failed to compute semantic alignment loss: {e}")
                import traceback
                traceback.print_exc()
                model_dict['semantic_alignment_loss'] = torch.tensor(0.0, device=support_images.device)

        else:
            # =====  =====
            use_eval_text = hasattr(self.args.TRAIN, "EVAL_TEXT") and self.args.TRAIN.EVAL_TEXT
            use_combine = hasattr(self.args.TRAIN, "COMBINE") and self.args.TRAIN.COMBINE

            support_features_raw, target_features_raw, _ = self.get_feats(
                support_images, target_images, support_labels)
            support_bs = support_features_raw.shape[0]
            target_bs = target_features_raw.shape[0]

            if use_eval_text or use_combine:
                # EVAL_TEXT / COMBINE forward  logits
                # get_feats  2  1  + super  1 
                model_dict = super(CNN_SEMANTIC_ALIGNMENT_FEW_SHOT, self).forward(inputs)
                # target context2
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
                # =====  OTAM context2(target)  =====
                # get_feats  1  super().forward()
                #  context_support  mid_layer
                feature_classification_in = torch.cat([support_features_raw, target_features_raw], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                class_text_logits = cos_sim(feature_classification, self.text_features_train) * self.scale

                #  query_embed  text_features
                if self.use_query_embed_eval:
                    context_support = self.query_embed_train.weight[support_real_class.long()].unsqueeze(1)
                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)

                #  context_support  mid_layer
                if self.use_mid_layer_eval:
                    context_support1 = self.mid_layer(context_support)
                else:
                    context_support1 = context_support

                # context_support  mid_layer2
                context_support2 = None
                if self.use_mid_layer2:
                    context_support2 = self.mid_layer2(context_support)

                # target context2OTAM 4
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
                    # 
                    support_enhanced_sem = self.context2(support_merged, support_merged, support_merged)
                    support_features_for_semantic = torch.stack([
                        support_enhanced_sem[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
                    #  context_support2  OTAM  context_support1
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
                    #  context2 
                    support_enhanced = self.context2(
                        support_features_raw, support_features_raw, support_features_raw)
                    support_enhanced_cls = torch.stack([
                        torch.mean(torch.index_select(support_enhanced, 0, extract_class_indices(support_labels, c)),
                                   dim=0)
                        for c in unique_labels])
                    support_features_for_semantic = torch.stack([
                        support_enhanced_cls[(unique_labels == support_labels[i]).nonzero(as_tuple=True)[0][0]]
                        for i in range(support_bs)])
                    #  context_support2  OTAM  context_support1
                    if self.use_mid_layer2_semantic_eval and self.use_mid_layer2:
                        support_features_for_semantic = self.context2(
                            torch.cat([support_features_raw, context_support2], dim=1),
                            torch.cat([support_features_raw, context_support2], dim=1),
                            torch.cat([support_features_raw, context_support2], dim=1)
                        )[:, :self.args.DATA.NUM_INPUT_FRAMES, :]

                # OTAM 
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

            # ----- 2target  target_features_ctx context2-----
            try:
                all_features = torch.cat([support_features_for_semantic, target_features_ctx], dim=0)

                all_class_names = []
                for i in range(support_bs):
                    all_class_names.append(self._get_class_name(int(support_real_class[i].item())))
                if 'real_target_labels' in inputs:
                    target_real_labels = inputs['real_target_labels']
                    for i in range(min(target_bs, len(target_real_labels))):
                        all_class_names.append(self._get_class_name(int(target_real_labels[i].item())))
                else:
                    for i in range(min(target_bs, support_bs)):
                        all_class_names.append(self._get_class_name(int(support_real_class[i].item())))
                if target_bs > len(all_class_names) - support_bs:
                    last_class = all_class_names[-1] if all_class_names else (
                        self.class_real_test[0] if self.class_real_test else "")
                    needed = target_bs - (len(all_class_names) - support_bs)
                    for _ in range(needed):
                        all_class_names.append(last_class)

                with torch.no_grad():
                    semantic_alignment_loss = self._compute_semantic_alignment_loss_multi_class(
                        all_features, all_class_names, debug=False)
                model_dict['semantic_alignment_loss'] = semantic_alignment_loss
            except Exception as e:
                print(f"Warning: Failed to compute semantic alignment loss: {e}")
                import traceback
                traceback.print_exc()
                model_dict['semantic_alignment_loss'] = torch.tensor(0.0, device=support_images.device)

            # + target_features_raw  get_feats
            model_dict = self._fuse_semantic_and_visual_probs_eval(
                inputs, model_dict, cached_target_features=target_features_raw)

        return model_dict

    # def loss(self, task_dict, model_dict):
    #     """
    #     few-shot + 
    #
    #     
    #     1. base_loss: Few-shot (support-query)
    #     2. semantic_alignment_loss:  (-)
    #
    #     Args:
    #         task_dict: 
    #         model_dict: 
    #
    #     Returns:
    #         dict: 
    #     """
    #     # few-shot
    #     base_loss = F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
    #
    #     # 
    #     semantic_loss = model_dict.get("semantic_alignment_loss", torch.tensor(0.0, device=base_loss.device))
    #
    #     # 
    #     total_loss = base_loss + self.semantic_loss_weight * semantic_loss
    #
    #     # batch size
    #     if hasattr(self.args.TRAIN, 'BATCH_SIZE'):
    #         total_loss = total_loss / self.args.TRAIN.BATCH_SIZE
    #
    #     # 
    #     return {
    #         'total_loss': total_loss,
    #         'frame_alignment_loss': base_loss,  # few-shot
    #         'semantic_loss': semantic_loss,
    #         'weighted_semantic_loss': self.semantic_loss_weight * semantic_loss
    #     }
