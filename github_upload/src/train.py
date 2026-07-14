"""
Training Loop - OneCycleLR (Super Convergence) ile
- Tüm modeller için ortak fonksiyon
- Train/Val tracking
- En iyi model kaydetme
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
import time
from pathlib import Path


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    """Bir epoch eğitim"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Train', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # OneCycleLR: HER batch'te step!
        
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.5f}'
        })
    
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validation"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Val', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total*100:.2f}%'
        })
    
    return total_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=20,
    max_lr=0.01,
    save_path='model.pth',
    device='cuda',
    pct_start=0.3
):
    """
    Modeli eğit + Super Convergence (OneCycleLR)
    
    Args:
        model: PyTorch modeli
        train_loader: Eğitim DataLoader
        val_loader: Validation DataLoader
        epochs: Epoch sayısı
        max_lr: Maksimum learning rate (OneCycleLR)
        save_path: En iyi modelin kaydedileceği path
        device: 'cuda' veya 'cpu'
        pct_start: Warm-up oranı (0.3 = %30'u yükseliş)
    
    Returns:
        history: dict (train_loss, train_acc, val_loss, val_acc, lrs)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # SGD + Momentum (OneCycleLR ile en iyi çalışır)
    optimizer = SGD(
        model.parameters(),
        lr=max_lr / 25,  # Initial LR (OneCycleLR otomatik artıracak)
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # OneCycleLR Scheduler (SUPER CONVERGENCE!)
    total_steps = epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    print(f"=" * 60)
    print(f"TRAINING BAŞLADI")
    print(f"=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Max LR: {max_lr}")
    print(f"Total steps: {total_steps}")
    print(f"Device: {device}")
    print(f"Save path: {save_path}")
    print(f"=" * 60)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lrs': []
    }
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lrs'].append(scheduler.get_last_lr()[0])
        
        epoch_time = time.time() - epoch_start
        
        # Print
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.5f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_path)
            print(f"   💾 Best model saved! Val Acc: {val_acc*100:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"EĞİTİM BİTTİ - Toplam süre: {total_time/60:.1f} dakika")
    print(f"En iyi Val Acc: {best_val_acc*100:.2f}%")
    print(f"Model kaydedildi: {save_path}")
    print(f"=" * 60)
    
    return history