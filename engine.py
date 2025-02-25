import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import torch.nn.functional as F
from collections import defaultdict

class AverageMeter:
    """Class to track metrics."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_metrics(pred, target):
    """Calculate evaluation metrics."""
    # Ensure inputs are boolean or 0/1 values
    pred = (pred > 0.5).float()
    target = target.float()
    
    # Calculate confusion matrix elements
    tp = torch.logical_and(pred, target).sum().float()
    tn = torch.logical_and(~pred.bool(), ~target.bool()).sum().float()
    fp = torch.logical_and(pred, ~target.bool()).sum().float()
    fn = torch.logical_and(~pred.bool(), target).sum().float()
    
    # Calculate metrics
    epsilon = 1e-7  # Prevent division by zero
    
    iou = tp / (tp + fp + fn + epsilon)
    f1_score = 2 * tp / (2 * tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    
    return {
        'iou': iou.item(),
        'f1_score': f1_score.item(),
        'accuracy': accuracy.item(),
        'sensitivity': sensitivity.item(),
        'specificity': specificity.item()
    }

def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer):
    """Train for one epoch."""
    losses = AverageMeter()
    metrics_sum = defaultdict(list)
    scaler = GradScaler(enabled=config.amp)
    
    # Ensure the model is in training mode
    model.train()
    
    # Add progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
    
    for iter, (images, targets) in enumerate(pbar):
        step += 1
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        with autocast(enabled=config.amp):
            outputs = model(images)
            
            # Handle main output and auxiliary outputs
            if isinstance(outputs, tuple):
                main_output, aux_outputs = outputs[0], outputs[1:]
                loss = criterion(main_output, aux_outputs, targets)
            else:
                loss = criterion(outputs, None, targets)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Update loss record
        losses.update(loss.item())
        
        # Calculate evaluation metrics
        if isinstance(outputs, tuple):
            pred = torch.sigmoid(main_output).detach()
        else:
            pred = torch.sigmoid(outputs).detach()
        
        metrics = calculate_metrics(pred, targets)
        for k, v in metrics.items():
            metrics_sum[k].append(v)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'IoU': f'{np.mean(metrics_sum["iou"]):.4f}'
        })
        
        # Log training information
        if writer is not None and step % config.print_interval == 0:
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], step)
            for k, v in metrics.items():
                writer.add_scalar(f'Metrics/{k}', v, step)
    
    pbar.close()
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items()}
    
    # Log training information
    log_info = f'Training Epoch {epoch}:\n' \
               f'Loss: {losses.avg:.4f}\n' \
               f'IoU: {avg_metrics["iou"]:.4f}\n' \
               f'F1/DSC: {avg_metrics["f1_score"]:.4f}\n' \
               f'Accuracy: {avg_metrics["accuracy"]:.4f}\n' \
               f'Sensitivity: {avg_metrics["sensitivity"]:.4f}\n' \
               f'Specificity: {avg_metrics["specificity"]:.4f}'
    logger.info(log_info)
    
    return losses.avg, step

def validate_one_epoch(val_loader, model, criterion, epoch, logger, config, writer=None):
    """Validate for one epoch."""
    losses = AverageMeter()
    metrics_sum = defaultdict(list)
    
    # Switch to evaluation mode
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} Validation')
        for i, (images, targets) in enumerate(pbar):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            # Use mixed precision inference
            with autocast(enabled=config.amp):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    main_output, aux_outputs = outputs[0], outputs[1:]
                    loss = criterion(main_output, aux_outputs, targets)
                    pred = torch.sigmoid(main_output)
                else:
                    loss = criterion(outputs, None, targets)
                    pred = torch.sigmoid(outputs)
            
            losses.update(loss.item())
            
            # Calculate evaluation metrics
            metrics = calculate_metrics(pred, targets)
            for k, v in metrics.items():
                metrics_sum[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'IoU': f'{np.mean(metrics_sum["iou"]):.4f}'
            })
            
            # Save prediction results
            if i % config.save_interval == 0:
                save_imgs(images, targets, pred, i, 
                         config.work_dir + 'outputs/', 
                         config.datasets)
            
            # Log validation information
            if writer is not None and i % config.print_interval == 0:
                writer.add_scalar('Loss/val', loss.item(), 
                                epoch * len(val_loader) + i)
        
        pbar.close()
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items()}
    
    # Log validation information
    log_info = f'Validation Results:\n' \
               f'Loss: {losses.avg:.4f}\n' \
               f'IoU: {avg_metrics["iou"]:.4f}\n' \
               f'F1/DSC: {avg_metrics["f1_score"]:.4f}\n' \
               f'Accuracy: {avg_metrics["accuracy"]:.4f}\n' \
               f'Sensitivity: {avg_metrics["sensitivity"]:.4f}\n' \
               f'Specificity: {avg_metrics["specificity"]:.4f}'
    logger.info(log_info)
    
    return losses.avg

def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    """Testing function."""
    model.eval()
    losses = AverageMeter()
    metrics_sum = defaultdict(list)
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for i, (images, targets) in enumerate(pbar):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
            # Use mixed precision inference
            with autocast(enabled=config.amp):
                outputs = model(images)
                if isinstance(outputs, tuple):
                    main_output, aux_outputs = outputs[0], outputs[1:]
                    loss = criterion(main_output, aux_outputs, targets)
                    pred = torch.sigmoid(main_output)
                else:
                    loss = criterion(outputs, None, targets)
                    pred = torch.sigmoid(outputs)
            
            losses.update(loss.item())
            
            # Calculate evaluation metrics
            metrics = calculate_metrics(pred, targets)
            for k, v in metrics.items():
                metrics_sum[k].append(v)
            
            # Save prediction results
            if i % config.save_interval == 0:
                save_imgs(images, targets, pred, i, 
                         config.work_dir + 'outputs/', 
                         config.datasets,
                         test_data_name=test_data_name)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'IoU': f'{np.mean(metrics_sum["iou"]):.4f}'
            })
        
        pbar.close()
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items()}
    
    # Log testing information
    log_info = f'Test Results:\n' \
               f'Loss: {losses.avg:.4f}\n' \
               f'IoU: {avg_metrics["iou"]:.4f}\n' \
               f'F1/DSC: {avg_metrics["f1_score"]:.4f}\n' \
               f'Accuracy: {avg_metrics["accuracy"]:.4f}\n' \
               f'Sensitivity: {avg_metrics["sensitivity"]:.4f}\n' \
               f'Specificity: {avg_metrics["specificity"]:.4f}'
    logger.info(log_info)
    
    return losses.avg