import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import shutil

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入我们的模块
from models.base.semantic_alignment_supervised_supervised_1 import CNN_SEMANTIC_ALIGNMENT_SUPERVISED
from datasets.supervised_dataset_supervised import create_supervised_dataloader


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """设置日志"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """创建模型"""
    # 创建模型参数对象
    class Args:
        def __init__(self, config):
            self.DATA = type('Data', (), config['DATA'])()
            # 创建默认的MODEL配置
            model_config = config.get('MODEL', {})
            if not model_config:
                model_config = {
                    'NAME': 'CNN_SEMANTIC_ALIGNMENT_SUPERVISED',
                    'BACKBONE_NAME': 'RN50',
                    'NUM_CLASSES': 51,
                    'DROPOUT': 0.5,
                    'FREEZE_BACKBONE': False
                }
            self.MODEL = type('Model', (), model_config)()
            self.TRAIN = type('Train', (), config['TRAIN'])()
            
            # 创建VIDEO配置（原始模型需要的）
            video_config = config.get('VIDEO', {})
            if not video_config:
                head_config = {
                    'NAME': 'CNN_SEMANTIC_ALIGNMENT_SUPERVISED',
                    'BACKBONE_NAME': 'RN50',
                    'NUM_CLASSES': 51
                }
                video_config = {
                    'HEAD': type('Head', (), head_config)()
                }
            else:
                # 如果配置中有VIDEO，确保HEAD是对象而不是字典
                if 'HEAD' in video_config and isinstance(video_config['HEAD'], dict):
                    head_config = video_config['HEAD']
                    video_config['HEAD'] = type('Head', (), head_config)()
            
            self.VIDEO = type('Video', (), video_config)()
            
            # 创建TEST配置（原始模型需要的）
            test_config = config.get('TEST', {})
            if not test_config:
                test_config = {
                    'CLASS_NAME': config['TRAIN'].get('CLASS_NAME', [])
                }
            self.TEST = type('Test', (), test_config)()
            
            # 使用扩展的类别名称
            self.CLASS_NAMES = config['TRAIN'].get('CLASS_NAME', [])
    
    args = Args(config)
    
    # 创建模型
    model = CNN_SEMANTIC_ALIGNMENT_SUPERVISED(args)
    
    return model, args


def create_dataloaders(config):
    """创建数据加载器"""
    data_config = config['DATA']
    
    # 训练数据加载器
    train_loader = create_supervised_dataloader(
        data_root=data_config['DATA_ROOT_DIR'],
        split='train',
        batch_size=data_config['BATCH_SIZE'],
        num_workers=data_config['NUM_WORKERS'],
        pin_memory=data_config.get('PIN_MEMORY', True),
        num_frames=data_config['NUM_INPUT_FRAMES'],
        sampling_rate=data_config['SAMPLING_RATE'],
        sampling_uniform=data_config['SAMPLING_UNIFORM'],
        crop_size=data_config['TRAIN_CROP_SIZE'],
        jitter_scales=data_config['TRAIN_JITTER_SCALES'],
        mean=data_config['MEAN'],
        std=data_config['STD'],
        class_names=config['TRAIN'].get('CLASS_NAME', [])
    )
    
    # 验证数据加载器
    val_loader = create_supervised_dataloader(
        data_root=data_config['DATA_ROOT_DIR'],
        split='val',
        batch_size=config['VAL']['BATCH_SIZE'],
        num_workers=config['VAL']['NUM_WORKERS'],
        pin_memory=data_config.get('PIN_MEMORY', True),
        num_frames=data_config['NUM_INPUT_FRAMES'],
        sampling_rate=data_config['SAMPLING_RATE'],
        sampling_uniform=data_config['SAMPLING_UNIFORM'],
        crop_size=data_config['TEST_CROP_SIZE'],
        jitter_scales=[data_config['TEST_SCALE'], data_config['TEST_SCALE']],
        mean=data_config['MEAN'],
        std=data_config['STD'],
        class_names=config['TRAIN'].get('CLASS_NAME', [])
    )
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model, config):
    """创建优化器和学习率调度器"""
    train_config = config['TRAIN']
    
    # 创建优化器
    if train_config['OPTIMIZER'].lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config['LEARNING_RATE'],
            momentum=train_config['MOMENTUM'],
            weight_decay=train_config['WEIGHT_DECAY'],
            nesterov=train_config.get('NESTEROV', False)
        )
    elif train_config['OPTIMIZER'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['LEARNING_RATE'],
            weight_decay=train_config['WEIGHT_DECAY']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {train_config['OPTIMIZER']}")
    
    # 创建学习率调度器
    if train_config['LR_SCHEDULER'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['EPOCHS'],
            eta_min=train_config.get('MIN_LR', 0.000001)
        )
    elif train_config['LR_SCHEDULER'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_config.get('STEP_SIZE', 30),
            gamma=train_config.get('GAMMA', 0.1)
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, device, epoch, config, logger):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_text_similarity_loss = 0.0
    total_semantic_loss = 0.0
    total_weighted_semantic_loss = 0.0
    correct = 0
    total = 0
    
    train_config = config['TRAIN']
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        videos = batch['video'].to(device)  # [batch_size, num_frames, C, H, W]
        labels = batch['label'].to(device)  # [batch_size]
        
        # 前向传播
        optimizer.zero_grad()
        
        # 在全监督学习中，支持集和目标集都是训练样本
        model_output = model(
            support_images=videos,
            target_images=videos,
            support_labels=labels,
            target_labels=labels
        )
        
        # 计算损失
        task_dict = {'target_labels': labels}
        loss_dict = model.loss(task_dict, model_output)
        loss = loss_dict['total_loss']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if train_config.get('GRAD_CLIP_NORM'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['GRAD_CLIP_NORM'])
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_text_similarity_loss += loss_dict['text_similarity_loss'].item()
        total_semantic_loss += loss_dict['semantic_loss'].item()
        total_weighted_semantic_loss += loss_dict['weighted_semantic_loss'].item()
        
        # 计算准确率
        _, predicted = torch.max(model_output['logits'], 1)
        # 确保预测结果和标签的批次大小匹配
        if predicted.shape[0] != labels.shape[0]:
            predicted = predicted[:labels.shape[0]]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 实时更新进度条（每个batch都更新）
        pbar.set_postfix({
            'Total': f'{loss.item():.4f}',
            'Text': f'{loss_dict["text_similarity_loss"].item():.4f}',
            'Semantic': f'{loss_dict["semantic_loss"].item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 每10个batch打印详细信息
        if batch_idx % 10 == 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)} - '
                       f'Total Loss: {loss.item():.4f}, '
                       f'Text Similarity Loss: {loss_dict["text_similarity_loss"].item():.4f}, '
                       f'Semantic Loss: {loss_dict["semantic_loss"].item():.4f}, '
                       f'Weighted Semantic Loss: {loss_dict["weighted_semantic_loss"].item():.4f}, '
                       f'Acc: {100. * correct / total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    avg_text_similarity_loss = total_text_similarity_loss / len(train_loader)
    avg_semantic_loss = total_semantic_loss / len(train_loader)
    avg_weighted_semantic_loss = total_weighted_semantic_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    logger.info(f'Epoch {epoch} - Total Loss: {avg_loss:.4f}, '
                f'Text Similarity Loss: {avg_text_similarity_loss:.4f}, '
                f'Semantic Loss: {avg_semantic_loss:.4f}, '
                f'Weighted Semantic Loss: {avg_weighted_semantic_loss:.4f}, '
                f'Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def validate(model, val_loader, device, epoch, logger):
    """验证模型"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation {epoch}')
        for batch_idx, batch in enumerate(pbar):
            videos = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            model_output = model(
                support_images=videos,
                target_images=videos,
                support_labels=labels,
                target_labels=labels
            )
            
            # 计算损失
            task_dict = {'target_labels': labels}
            loss_dict = model.loss(task_dict, model_output)
            loss = loss_dict['total_loss']
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(model_output['logits'], 1)
            # 确保预测结果和标签的批次大小匹配
            if predicted.shape[0] != labels.shape[0]:
                predicted = predicted[:labels.shape[0]]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 实时更新验证进度条
            pbar.set_postfix({
                'Val_Total': f'{loss.item():.4f}',
                'Val_Text': f'{loss_dict["text_similarity_loss"].item():.4f}',
                'Val_Semantic': f'{loss_dict["semantic_loss"].item():.4f}',
                'Val_Acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    logger.info(f'Epoch {epoch} - Val Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, output_dir, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存最新检查点
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳检查点
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f'New best model saved with accuracy: {best_acc:.2f}%')


def main():
    # 启用异常检测
    torch.autograd.set_detect_anomaly(True)
    
    parser = argparse.ArgumentParser(description='Train Semantic Alignment Model (Supervised)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子
    if 'SEED' in config:
        torch.manual_seed(config['SEED'])
        np.random.seed(config['SEED'])
    
    # 设置日志
    output_dir = config['OUTPUT_DIR']
    logger = setup_logging(output_dir, config.get('LOG', {}).get('LEVEL', 'INFO'))
    
    logger.info(f'Starting training with config: {args.config}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Device: {device}')
    
    # 创建模型
    model, model_args = create_model(config)
    model = model.to(device)
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(config)
    
    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        logger.info(f'Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%')
    
    # 训练循环
    train_config = config['TRAIN']
    
    for epoch in range(start_epoch, train_config['EPOCHS']):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch, config, logger
        )
        
        # 验证
        if epoch % train_config.get('VAL_FREQ', 5) == 0:
            val_loss, val_acc = validate(model, val_loader, device, epoch, logger)
            
            # 保存最佳模型
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc, output_dir, is_best
            )
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        # 定期保存
        if epoch % train_config.get('SAVE_FREQ', 10) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, output_dir)
    
    logger.info(f'Training completed! Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
