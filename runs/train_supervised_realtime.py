#!/usr/bin/env python3
"""
增强版训练脚本 - 实时监控acc和loss变化
"""

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml
import logging
from tqdm import tqdm
import time
from datetime import datetime

def create_realtime_monitor():
    """创建实时监控器"""
    class RealtimeMonitor:
        def __init__(self):
            self.loss_history = []
            self.acc_history = []
            self.batch_times = []
            self.start_time = time.time()
            
        def update(self, loss, acc, batch_idx):
            """更新监控数据"""
            self.loss_history.append(loss)
            self.acc_history.append(acc)
            self.batch_times.append(time.time())
            
            # 计算趋势
            if len(self.loss_history) >= 10:
                recent_losses = self.loss_history[-10:]
                recent_accs = self.acc_history[-10:]
                
                loss_trend = "📈" if recent_losses[-1] > recent_losses[0] else "📉"
                acc_trend = "📈" if recent_accs[-1] > recent_accs[0] else "📉"
                
                print(f"\n🔄 Batch {batch_idx:4d} | "
                      f"Loss: {loss:.4f} {loss_trend} | "
                      f"Acc: {acc:.2f}% {acc_trend} | "
                      f"Time: {time.time() - self.start_time:.1f}s")
        
        def get_summary(self):
            """获取训练摘要"""
            if not self.loss_history:
                return "No data yet"
            
            avg_loss = sum(self.loss_history) / len(self.loss_history)
            avg_acc = sum(self.acc_history) / len(self.acc_history)
            best_acc = max(self.acc_history)
            best_loss = min(self.loss_history)
            
            return (f"📊 Summary: Avg Loss: {avg_loss:.4f}, "
                   f"Avg Acc: {avg_acc:.2f}%, "
                   f"Best Acc: {best_acc:.2f}%, "
                   f"Best Loss: {best_loss:.4f}")
    
    return RealtimeMonitor()

def enhanced_train_epoch(model, train_loader, optimizer, device, epoch, config, logger):
    """增强版训练函数 - 实时监控"""
    model.train()
    
    total_loss = 0.0
    total_text_loss = 0.0
    total_semantic_loss = 0.0
    correct = 0
    total = 0
    
    # 创建实时监控器
    monitor = create_realtime_monitor()
    
    print(f"\n🚀 Starting Epoch {epoch}")
    print("=" * 80)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        batch_start_time = time.time()
        
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
        loss = model.loss(task_dict, model_output)
        
        # 获取各个损失组件
        text_loss = model_output.get('text_loss', torch.tensor(0.0))
        semantic_loss = model_output.get('semantic_loss', torch.tensor(0.0))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新统计
        total_loss += loss.item()
        total_text_loss += text_loss.item()
        total_semantic_loss += semantic_loss.item()
        
        # 计算准确率
        _, predicted = torch.max(model_output['logits'], 1)
        if predicted.shape[0] != labels.shape[0]:
            predicted = predicted[:labels.shape[0]]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        current_acc = 100. * correct / total
        
        # 实时更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'Speed': f'{1/(time.time() - batch_start_time):.1f}it/s'
        })
        
        # 实时监控更新
        monitor.update(loss.item(), current_acc, batch_idx)
        
        # 每50个batch打印详细信息
        if batch_idx % 50 == 0 and batch_idx > 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)} - '
                       f'Loss: {loss.item():.4f}, '
                       f'Text Loss: {text_loss.item():.4f}, '
                       f'Semantic Loss: {semantic_loss.item():.4f}, '
                       f'Acc: {current_acc:.2f}%')
    
    # 打印训练摘要
    print("\n" + "=" * 80)
    print(monitor.get_summary())
    print("=" * 80)
    
    avg_loss = total_loss / len(train_loader)
    avg_text_loss = total_text_loss / len(train_loader)
    avg_semantic_loss = total_semantic_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    logger.info(f'Epoch {epoch} - Train Loss: {avg_loss:.4f}, '
                f'Text Loss: {avg_text_loss:.4f}, '
                f'Semantic Loss: {avg_semantic_loss:.4f}, '
                f'Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy

if __name__ == "__main__":
    print("🎯 Enhanced Training Script with Realtime Monitoring")
    print("This script provides real-time loss and accuracy monitoring")
    print("Use this for detailed training observation")
