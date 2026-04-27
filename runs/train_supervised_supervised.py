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

# 娣诲姞椤圭洰鏍圭洰褰曞埌璺緞
sys.path.append('/home/u/SF-CLIP')
sys.path.append('/home/u/SF-CLIP/supervised_learning')

# 娣诲姞褰撳墠鑴氭湰鐨勭埗鐩綍鍒拌矾寰?current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 瀵煎叆鎴戜滑鐨勬ā鍧?from models.base.semantic_alignment_supervised_supervised_1 import CNN_SEMANTIC_ALIGNMENT_SUPERVISED
from datasets.supervised_dataset_supervised import create_supervised_dataloader


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """璁剧疆鏃ュ織"""
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
    """鍔犺浇閰嶇疆鏂囦欢"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """鍒涘缓妯″瀷"""
    # 鍒涘缓妯″瀷鍙傛暟瀵硅薄
    class Args:
        def __init__(self, config):
            self.DATA = type('Data', (), config['DATA'])()
            # 鍒涘缓榛樿鐨凪ODEL閰嶇疆
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
            
            # 鍒涘缓VIDEO閰嶇疆锛堝師濮嬫ā鍨嬮渶瑕佺殑锛?            video_config = config.get('VIDEO', {})
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
                # 濡傛灉閰嶇疆涓湁VIDEO锛岀‘淇滺EAD鏄璞¤€屼笉鏄瓧鍏?                if 'HEAD' in video_config and isinstance(video_config['HEAD'], dict):
                    head_config = video_config['HEAD']
                    video_config['HEAD'] = type('Head', (), head_config)()
            
            self.VIDEO = type('Video', (), video_config)()
            
            # 鍒涘缓TEST閰嶇疆锛堝師濮嬫ā鍨嬮渶瑕佺殑锛?            test_config = config.get('TEST', {})
            if not test_config:
                test_config = {
                    'CLASS_NAME': config['TRAIN'].get('CLASS_NAME', [])
                }
            self.TEST = type('Test', (), test_config)()
            
            # 浣跨敤鎵╁睍鐨勭被鍒悕绉?            self.CLASS_NAMES = config['TRAIN'].get('CLASS_NAME', [])
    
    args = Args(config)
    
    # 鍒涘缓妯″瀷
    model = CNN_SEMANTIC_ALIGNMENT_SUPERVISED(args)
    
    return model, args


def create_dataloaders(config):
    """鍒涘缓鏁版嵁鍔犺浇鍣?""
    data_config = config['DATA']
    
    # 璁粌鏁版嵁鍔犺浇鍣?    train_loader = create_supervised_dataloader(
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
    
    # 楠岃瘉鏁版嵁鍔犺浇鍣?    val_loader = create_supervised_dataloader(
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
    """鍒涘缓浼樺寲鍣ㄥ拰瀛︿範鐜囪皟搴﹀櫒"""
    train_config = config['TRAIN']
    
    # 鍒涘缓浼樺寲鍣?    if train_config['OPTIMIZER'].lower() == 'sgd':
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
    
    # 鍒涘缓瀛︿範鐜囪皟搴﹀櫒
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
    """璁粌涓€涓猠poch"""
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
        
        # 鍓嶅悜浼犳挱
        optimizer.zero_grad()
        
        # 鍦ㄥ叏鐩戠潱瀛︿範涓紝鏀寔闆嗗拰鐩爣闆嗛兘鏄缁冩牱鏈?        model_output = model(
            support_images=videos,
            target_images=videos,
            support_labels=labels,
            target_labels=labels
        )
        
        # 璁＄畻鎹熷け
        task_dict = {'target_labels': labels}
        loss_dict = model.loss(task_dict, model_output)
        loss = loss_dict['total_loss']
        
        # 鍙嶅悜浼犳挱
        loss.backward()
        
        # 姊害瑁佸壀
        if train_config.get('GRAD_CLIP_NORM'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['GRAD_CLIP_NORM'])
        
        optimizer.step()
        
        # 缁熻
        total_loss += loss.item()
        total_text_similarity_loss += loss_dict['text_similarity_loss'].item()
        total_semantic_loss += loss_dict['semantic_loss'].item()
        total_weighted_semantic_loss += loss_dict['weighted_semantic_loss'].item()
        
        # 璁＄畻鍑嗙‘鐜?        _, predicted = torch.max(model_output['logits'], 1)
        # 纭繚棰勬祴缁撴灉鍜屾爣绛剧殑鎵规澶у皬鍖归厤
        if predicted.shape[0] != labels.shape[0]:
            predicted = predicted[:labels.shape[0]]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 瀹炴椂鏇存柊杩涘害鏉★紙姣忎釜batch閮芥洿鏂帮級
        pbar.set_postfix({
            'Total': f'{loss.item():.4f}',
            'Text': f'{loss_dict["text_similarity_loss"].item():.4f}',
            'Semantic': f'{loss_dict["semantic_loss"].item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 姣?0涓猙atch鎵撳嵃璇︾粏淇℃伅
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
    """楠岃瘉妯″瀷"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation {epoch}')
        for batch_idx, batch in enumerate(pbar):
            videos = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            # 鍓嶅悜浼犳挱
            model_output = model(
                support_images=videos,
                target_images=videos,
                support_labels=labels,
                target_labels=labels
            )
            
            # 璁＄畻鎹熷け
            task_dict = {'target_labels': labels}
            loss_dict = model.loss(task_dict, model_output)
            loss = loss_dict['total_loss']
            
            total_loss += loss.item()
            
            # 璁＄畻鍑嗙‘鐜?            _, predicted = torch.max(model_output['logits'], 1)
            # 纭繚棰勬祴缁撴灉鍜屾爣绛剧殑鎵规澶у皬鍖归厤
            if predicted.shape[0] != labels.shape[0]:
                predicted = predicted[:labels.shape[0]]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 瀹炴椂鏇存柊楠岃瘉杩涘害鏉?            pbar.set_postfix({
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
    """淇濆瓨妫€鏌ョ偣"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 淇濆瓨鏈€鏂版鏌ョ偣
    checkpoint_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 淇濆瓨鏈€浣虫鏌ョ偣
    if is_best:
        best_path = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f'New best model saved with accuracy: {best_acc:.2f}%')


def main():
    # 鍚敤寮傚父妫€娴?    torch.autograd.set_detect_anomaly(True)
    
    parser = argparse.ArgumentParser(description='Train Semantic Alignment Model (Supervised)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # 鍔犺浇閰嶇疆
    config = load_config(args.config)
    
    # 璁剧疆璁惧
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 璁剧疆闅忔満绉嶅瓙
    if 'SEED' in config:
        torch.manual_seed(config['SEED'])
        np.random.seed(config['SEED'])
    
    # 璁剧疆鏃ュ織
    output_dir = config['OUTPUT_DIR']
    logger = setup_logging(output_dir, config.get('LOG', {}).get('LEVEL', 'INFO'))
    
    logger.info(f'Starting training with config: {args.config}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Device: {device}')
    
    # 鍒涘缓妯″瀷
    model, model_args = create_model(config)
    model = model.to(device)
    
    # 鍒涘缓鏁版嵁鍔犺浇鍣?    train_loader, val_loader = create_dataloaders(config)
    
    # 鍒涘缓浼樺寲鍣ㄥ拰璋冨害鍣?    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # 鎭㈠璁粌
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
    
    # 璁粌寰幆
    train_config = config['TRAIN']
    
    for epoch in range(start_epoch, train_config['EPOCHS']):
        # 璁粌
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch, config, logger
        )
        
        # 楠岃瘉
        if epoch % train_config.get('VAL_FREQ', 5) == 0:
            val_loss, val_acc = validate(model, val_loader, device, epoch, logger)
            
            # 淇濆瓨鏈€浣虫ā鍨?            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc, output_dir, is_best
            )
        
        # 鏇存柊瀛︿範鐜?        if scheduler:
            scheduler.step()
        
        # 瀹氭湡淇濆瓨
        if epoch % train_config.get('SAVE_FREQ', 10) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, output_dir)
    
    logger.info(f'Training completed! Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
