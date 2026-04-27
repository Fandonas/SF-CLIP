#!/usr/bin/env python3
"""
澧炲己鐗堣缁冭剼鏈?- 瀹炴椂鐩戞帶acc鍜宭oss鍙樺寲
"""

import sys
import os
sys.path.append('/home/u/SF-CLIP')

import torch
import yaml
import logging
from tqdm import tqdm
import time
from datetime import datetime

def create_realtime_monitor():
    """鍒涘缓瀹炴椂鐩戞帶鍣?""
    class RealtimeMonitor:
        def __init__(self):
            self.loss_history = []
            self.acc_history = []
            self.batch_times = []
            self.start_time = time.time()
            
        def update(self, loss, acc, batch_idx):
            """鏇存柊鐩戞帶鏁版嵁"""
            self.loss_history.append(loss)
            self.acc_history.append(acc)
            self.batch_times.append(time.time())
            
            # 璁＄畻瓒嬪娍
            if len(self.loss_history) >= 10:
                recent_losses = self.loss_history[-10:]
                recent_accs = self.acc_history[-10:]
                
                loss_trend = "馃搱" if recent_losses[-1] > recent_losses[0] else "馃搲"
                acc_trend = "馃搱" if recent_accs[-1] > recent_accs[0] else "馃搲"
                
                print(f"\n馃攧 Batch {batch_idx:4d} | "
                      f"Loss: {loss:.4f} {loss_trend} | "
                      f"Acc: {acc:.2f}% {acc_trend} | "
                      f"Time: {time.time() - self.start_time:.1f}s")
        
        def get_summary(self):
            """鑾峰彇璁粌鎽樿"""
            if not self.loss_history:
                return "No data yet"
            
            avg_loss = sum(self.loss_history) / len(self.loss_history)
            avg_acc = sum(self.acc_history) / len(self.acc_history)
            best_acc = max(self.acc_history)
            best_loss = min(self.loss_history)
            
            return (f"馃搳 Summary: Avg Loss: {avg_loss:.4f}, "
                   f"Avg Acc: {avg_acc:.2f}%, "
                   f"Best Acc: {best_acc:.2f}%, "
                   f"Best Loss: {best_loss:.4f}")
    
    return RealtimeMonitor()

def enhanced_train_epoch(model, train_loader, optimizer, device, epoch, config, logger):
    """澧炲己鐗堣缁冨嚱鏁?- 瀹炴椂鐩戞帶"""
    model.train()
    
    total_loss = 0.0
    total_text_loss = 0.0
    total_semantic_loss = 0.0
    correct = 0
    total = 0
    
    # 鍒涘缓瀹炴椂鐩戞帶鍣?    monitor = create_realtime_monitor()
    
    print(f"\n馃殌 Starting Epoch {epoch}")
    print("=" * 80)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        batch_start_time = time.time()
        
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
        loss = model.loss(task_dict, model_output)
        
        # 鑾峰彇鍚勪釜鎹熷け缁勪欢
        text_loss = model_output.get('text_loss', torch.tensor(0.0))
        semantic_loss = model_output.get('semantic_loss', torch.tensor(0.0))
        
        # 鍙嶅悜浼犳挱
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 鏇存柊缁熻
        total_loss += loss.item()
        total_text_loss += text_loss.item()
        total_semantic_loss += semantic_loss.item()
        
        # 璁＄畻鍑嗙‘鐜?        _, predicted = torch.max(model_output['logits'], 1)
        if predicted.shape[0] != labels.shape[0]:
            predicted = predicted[:labels.shape[0]]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        current_acc = 100. * correct / total
        
        # 瀹炴椂鏇存柊杩涘害鏉?        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'Speed': f'{1/(time.time() - batch_start_time):.1f}it/s'
        })
        
        # 瀹炴椂鐩戞帶鏇存柊
        monitor.update(loss.item(), current_acc, batch_idx)
        
        # 姣?0涓猙atch鎵撳嵃璇︾粏淇℃伅
        if batch_idx % 50 == 0 and batch_idx > 0:
            logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)} - '
                       f'Loss: {loss.item():.4f}, '
                       f'Text Loss: {text_loss.item():.4f}, '
                       f'Semantic Loss: {semantic_loss.item():.4f}, '
                       f'Acc: {current_acc:.2f}%')
    
    # 鎵撳嵃璁粌鎽樿
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
    print("馃幆 Enhanced Training Script with Realtime Monitoring")
    print("This script provides real-time loss and accuracy monitoring")
    print("Use this for detailed training observation")
