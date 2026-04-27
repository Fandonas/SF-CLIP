import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import random
from torchvision import transforms


class SupervisedVideoDataset(Dataset):
    """
    全监督学习的视频数据集
    
    与few-shot数据集的主要区别：
    1. 不需要支持集和查询集的划分
    2. 每个样本都是独立的训练样本
    3. 使用标准的训练/验证/测试划分
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 num_frames: int = 8,
                 sampling_rate: int = 50,
                 sampling_uniform: bool = False,
                 crop_size: int = 224,
                 jitter_scales: List[int] = [256, 256],
                 mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
                 std: List[float] = [0.26862954, 0.26130258, 0.27577711],
                 class_names: Optional[List[str]] = None):
        """
        Args:
            data_root: 数据根目录
            split: 数据集划分 ('train', 'val', 'test')
            num_frames: 每个视频采样的帧数
            sampling_rate: 采样率
            sampling_uniform: 是否均匀采样
            crop_size: 裁剪尺寸
            jitter_scales: 随机缩放范围
            mean: 归一化均值
            std: 归一化标准差
            class_names: 类别名称列表
        """
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.sampling_uniform = sampling_uniform
        self.crop_size = crop_size
        self.jitter_scales = jitter_scales
        self.mean = mean
        self.std = std
        
        # 加载类别信息
        self.extended_class_names = class_names or self._load_class_names()
        self.simple_class_names = self._get_simple_class_names()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.simple_class_names)}
        
        # 创建扩展名称到简单名称的映射
        self.extended_to_simple = self._create_name_mapping()
        
        # 加载视频列表
        self.video_list = self._load_video_list()
        
        # 数据增强
        self.transform = self._get_transform()
        
    def _load_class_names(self) -> List[str]:
        """加载扩展的类别名称"""
        # 返回扩展的类别名称（从配置文件传入）
        return []
    
    def _get_simple_class_names(self) -> List[str]:
        """获取简单的类别名称"""
        simple_class_names = [
            'brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs',
            'dive', 'draw_sword', 'dribble', 'drink', 'eat', 'fall_floor', 'fencing',
            'flic_flac', 'golf', 'handstand', 'hit', 'hug', 'jump', 'kick', 'kick_ball',
            'kiss', 'laugh', 'pick', 'pour', 'pullup', 'punch', 'push', 'pushup',
            'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball', 'shoot_bow',
            'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault', 'stand',
            'swing_baseball', 'sword', 'sword_exercise', 'talk', 'throw', 'turn',
            'walk', 'wave'
        ]
        return simple_class_names
    
    def _create_name_mapping(self) -> Dict[str, str]:
        """创建扩展名称到简单名称的映射"""
        # 扩展名称到简单名称的映射（完整的51个HMDB51类别）
        mapping = {
            'Running fingers or brush through hair repeatedly, then moving from scalp to ends in smooth strokes': 'brush_hair',
            'Placing hands on the ground sideways, then rotating the body in a circular motion while maintaining contact': 'cartwheel',
            'Following the trajectory of an airborne object, then hands successfully intercept and secure it': 'catch',
            'Placing food in mouth, then jaw moves rhythmically to process and break down the substance': 'chew',
            'Bringing palms together with force, then creating a sharp sound that echoes through the space': 'clap',
            'Reaching upward to find secure grips, then pushing with feet to lift body weight higher': 'climb',
            'Stepping onto the first level, then continuing to ascend each subsequent step to reach the top': 'climb_stairs',
            'Positioning at the platform edge, then launching headfirst into the water below': 'dive',
            'Grasping the sword handle firmly, then drawing the blade free from its protective sheath': 'draw_sword',
            'Dropping the ball to the ground, then it bounces back up to be caught and released again': 'dribble',
            'Raising the container to mouth level, then allowing liquid to flow smoothly into the opening': 'drink',
            'Bringing food to the mouth, then teeth and tongue work together to consume the substance': 'eat',
            'Losing balance and control, then the body falls to the ground surface': 'fall_floor',
            'Holding the sword in combat stance, then executing precise thrusting and parrying movements': 'fencing',
            'Beginning in upright position, then arching backward to complete a full backward somersault': 'flic_flac',
            'Positioning the club behind the ball, then swinging forward to strike and send it flying': 'golf',
            'Planting hands firmly on ground, then raising legs to balance entire body weight upside down': 'handstand',
            'Drawing arm back to build power, then releasing a forceful strike toward the target': 'hit',
            'Opening arms wide, then closing them around another in a warm, protective embrace': 'hug',
            'Bending knees to prepare, then exploding upward with leg power to reach maximum height': 'jump',
            'Lifting the leg high, then snapping it forward to strike with precise force': 'kick',
            'Approaching the ball with foot raised, then making contact to send the ball flying': 'kick_ball',
            'Moving face close to another, then lips meet in a brief, tender connection': 'kiss',
            'Opening the mouth wide, then producing rhythmic sounds of joy and amusement': 'laugh',
            'Reaching toward the object, then fingers close around it to lift the object away': 'pick',
            'Tilting the container at an angle, then liquid flows smoothly into the receiving vessel': 'pour',
            'Gripping the horizontal bar, then pulling with arms until the body rises above the bar': 'pullup',
            'Drawing arm back to build power, then releasing a forceful punch toward the target': 'punch',
            'Placing hands against the object, then applying steady pressure to push the object away': 'push',
            'Placing hands on the ground, then lowering and raising the body through arm strength': 'pushup',
            'Settling into the bicycle seat, then using legs to pedal and hands to guide direction': 'ride_bike',
            'Mounting the horses back, then using legs and reins to direct the animals movement': 'ride_horse',
            'Moving legs in rapid succession, then covering distance quickly with each forward step': 'run',
            'Extending hand forward, then grasping anothers hand and moving it in greeting': 'shake_hands',
            'Aiming the ball toward the basket, then releasing it with precise trajectory and force': 'shoot_ball',
            'Drawing the bowstring back with tension, then releasing to send arrow toward target': 'shoot_bow',
            'Aiming the weapon at the target, then pulling the trigger to release the projectile': 'shoot_gun',
            'Bending at the knees and waist, then the body settles into a seated position': 'sit',
            'Lying flat on the surface, then contracting core muscles to raise torso upward': 'situp',
            'Curving the lips upward, then eyes crinkle to express happiness and warmth': 'smile',
            'Holding the smoking device, then inhaling and exhaling smoke in controlled rhythm': 'smoke',
            'Tucking the head and rolling forward, then completing a full rotation on the ground': 'somersault',
            'Rising from a lower position, then standing upright until the body is balanced on feet': 'stand',
            'Cocking the arm back behind the body, then whipping forward to launch the object': 'swing_baseball',
            'Wielding the sword with purpose, then moving it through various combat-ready positions': 'sword',
            'Holding the sword steadily, then executing a series of coordinated training movements': 'sword_exercise',
            'Opening the mouth to form words, then vocal cords produce sounds for communication': 'talk',
            'Cocking the arm back behind the body, then whipping forward to launch the object': 'throw',
            'Rotating around a central axis, then continuing the circular motion in smooth rhythm': 'turn',
            'Lifting one foot from ground, then placing it forward to advance the bodys position': 'walk',
            'Raising hand above the head, then moving it back and forth to create a visible signal': 'wave'
        }
        return mapping
    
    def _load_video_list(self) -> List[Dict]:
        """加载视频列表并根据split进行划分"""
        all_video_list = []
        
        # 从目录结构加载（HMDB51数据集结构）
        videos_dir = os.path.join(self.data_root, 'videos')
        print(f"数据根目录: {self.data_root}")
        print(f"视频目录: {videos_dir}")
        print(f"视频目录存在: {os.path.exists(videos_dir)}")
        print(f"简单类别名称: {self.simple_class_names[:5]}...")
        
        if os.path.exists(videos_dir):
            for simple_class_name in self.simple_class_names:
                class_dir = os.path.join(videos_dir, simple_class_name)
                if os.path.exists(class_dir):
                    video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
                    print(f"类别 {simple_class_name}: {len(video_files)} 个视频文件")
                    for video_file in video_files:
                        all_video_list.append({
                            'video_path': os.path.join(class_dir, video_file),
                            'class_name': simple_class_name,
                            'class_idx': self.class_to_idx[simple_class_name]
                        })
                else:
                    print(f"类别目录不存在: {class_dir}")
        
        print(f"总共加载了 {len(all_video_list)} 个视频文件")
        
        # 根据split进行数据划分（70% train, 15% val, 15% test）
        video_list = self._split_videos_by_class(all_video_list)
        print(f"Split '{self.split}': {len(video_list)} 个视频文件")
        
        return video_list
    
    def _split_videos_by_class(self, all_videos: List[Dict]) -> List[Dict]:
        """按类别进行数据划分"""
        # 按类别分组
        videos_by_class = {}
        for video in all_videos:
            class_name = video['class_name']
            if class_name not in videos_by_class:
                videos_by_class[class_name] = []
            videos_by_class[class_name].append(video)
        
        # 为每个类别进行划分
        split_videos = []
        for class_name, class_videos in videos_by_class.items():
            # 随机打乱
            random.shuffle(class_videos)
            
            # 计算划分点
            total_count = len(class_videos)
            train_count = int(total_count * 0.7)
            val_count = int(total_count * 0.15)
            
            if self.split == 'train':
                split_videos.extend(class_videos[:train_count])
            elif self.split == 'val':
                split_videos.extend(class_videos[train_count:train_count + val_count])
            elif self.split == 'test':
                split_videos.extend(class_videos[train_count + val_count:])
            else:
                raise ValueError(f"Unknown split: {self.split}")
        
        return split_videos
    
    def _get_transform(self):
        """获取数据变换"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.jitter_scales),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.jitter_scales),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
    
    def _sample_frames(self, video_path: str) -> np.ndarray:
        """从视频中采样帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            # 如果视频帧数不足，重复采样
            frame_indices = list(range(total_frames)) * (self.num_frames // total_frames + 1)
            frame_indices = frame_indices[:self.num_frames]
        else:
            if self.sampling_uniform:
                # 均匀采样
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                # 随机采样
                frame_indices = sorted(random.sample(range(total_frames), self.num_frames))
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # 如果读取失败，使用最后一帧
                if frames:
                    frames.append(frames[-1])
                else:
                    # 如果没有任何帧，创建黑色帧
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        return np.array(frames)
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, idx):
        video_info = self.video_list[idx]
        
        # 采样视频帧
        frames = self._sample_frames(video_info['video_path'])
        
        # 应用变换
        transformed_frames = []
        for frame in frames:
            transformed_frame = self.transform(frame)
            transformed_frames.append(transformed_frame)
        
        # 堆叠为张量 [num_frames, C, H, W]
        video_tensor = torch.stack(transformed_frames)
        
        return {
            'video': video_tensor,
            'label': video_info['class_idx'],
            'class_name': video_info['class_name']
        }


def create_supervised_dataloader(data_root: str,
                                split: str = 'train',
                                batch_size: int = 32,
                                num_workers: int = 4,
                                pin_memory: bool = True,
                                **dataset_kwargs) -> DataLoader:
    """
    创建全监督学习的数据加载器
    
    Args:
        data_root: 数据根目录
        split: 数据集划分
        batch_size: 批次大小
        num_workers: 工作进程数
        **dataset_kwargs: 传递给数据集的其他参数
        
    Returns:
        DataLoader: 数据加载器
    """
    dataset = SupervisedVideoDataset(data_root=data_root, split=split, **dataset_kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train')
    )
    
    return dataloader
