import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

def set_seed(seed):
    """Set random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def get_logger(name, log_dir):
    """
    Create a logger
    Args:
        name(str): Logger name
        log_dir(str): Path to save logs
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(
        info_name,
        when='D',
        encoding='utf-8'
    )
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    return logger

def log_config_info(config, logger):
    """Log configuration information."""
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if not k.startswith('_'):
            logger.info(f'{k}: {v}')

# Loss function related classes
class BCELoss(nn.Module):
    """Binary Cross Entropy Loss"""
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, pred, aux_outputs, target):
        if aux_outputs is None:
            return self.bceloss(pred, target)
        
        # Calculate main output loss
        main_loss = self.bceloss(pred, target)
        
        # Calculate auxiliary output loss
        aux_loss = sum(self.bceloss(aux, target) for aux in aux_outputs)
        
        return main_loss + 0.4 * aux_loss

class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, aux_outputs, target):
        if aux_outputs is None:
            return self._dice_loss(pred, target)
        
        # Calculate main output loss
        main_loss = self._dice_loss(pred, target)
        
        # Calculate auxiliary output loss
        aux_loss = sum(self._dice_loss(aux, target) for aux in aux_outputs)
        
        return main_loss + 0.4 * aux_loss

    def _dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1.0
        
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)
        
        intersection = (pred_flat * target_flat).sum(1)
        unionset = pred_flat.sum(1) + target_flat.sum(1)
        loss = 1 - (2 * intersection + smooth) / (unionset + smooth)
        
        return loss.mean()

class GT_BceDiceLoss(nn.Module):
    """Combined loss function, supports multi-output of CDIUNet"""
    def __init__(self, wb=1.0, wd=1.0):
        super(GT_BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, main_output, aux_outputs, target):
        if aux_outputs is None:
            bce_loss = self.bce(main_output, None, target)
            dice_loss = self.dice(main_output, None, target)
        else:
            bce_loss = self.bce(main_output, aux_outputs, target)
            dice_loss = self.dice(main_output, aux_outputs, target)
        
        return self.wb * bce_loss + self.wd * dice_loss

# Data transformation related classes
class myTransform:
    """Base transformation class"""
    def __init__(self):
        pass

    def __call__(self, data):
        raise NotImplementedError

class myToTensor(myTransform):
    """Convert to Tensor"""
    def __call__(self, data):
        image, mask = data
        
        # Process image
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        
        # Process mask
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        mask = torch.from_numpy(mask.transpose((2, 0, 1)))
        
        return image, mask

class myResize(myTransform):
    """Resize"""
    def __init__(self, size_h, size_w):
        self.size_h = size_h
        self.size_w = size_w
        
    def __call__(self, data):
        image, mask = data
        return (TF.resize(image, [self.size_h, self.size_w]), 
                TF.resize(mask, [self.size_h, self.size_w]))

class myRandomHorizontalFlip(myTransform):
    """Random horizontal flip"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        return image, mask

class myRandomVerticalFlip(myTransform):
    """Random vertical flip"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        return image, mask

class myRandomRotation(myTransform):
    """Random rotation"""
    def __init__(self, p=0.5, degree=[-30, 30]):
        self.p = p
        self.degree = degree
        
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            return TF.rotate(image, angle), TF.rotate(mask, angle)
        return image, mask

class myNormalize(myTransform):
    """Normalization"""
    def __init__(self, data_name, train=True):
        self.mean = 153.2975
        self.std = 29.364

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / (self.std + 1e-6)
        img_normalized = ((img_normalized - np.min(img_normalized)) / 
                        (np.max(img_normalized) - np.min(img_normalized) + 1e-6))
        return img_normalized, msk

def save_imgs(images, targets, pred, batch_idx, save_path, threshold=0.5):
    """Save images, targets, and prediction results visualization
    Args:
        images: Input images
        targets: Target masks
        pred: Model prediction results
        batch_idx: Batch index
        save_path: Save path
        threshold: Binarization threshold, default is 0.5
    """
    # Ensure threshold is a float
    if isinstance(threshold, str):
        threshold = 0.5  # Use default value if a string is passed
    else:
        threshold = float(threshold)
    
    # Convert to NumPy array and ensure data type
    pred = pred.detach().cpu().numpy()
    pred = np.squeeze(pred)  # Remove batch and channel dimensions
    pred = pred.astype(np.float32)  # Ensure using float32 type
    pred_binary = (pred >= threshold).astype(np.uint8)
    
    if torch.is_tensor(images):
        images = images.detach().cpu().numpy()
        images = np.transpose(images[0], (1, 2, 0))  # Move channel dimension to the last
        images = images.astype(np.float32)  # Ensure using float32 type
    
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
        targets = np.squeeze(targets)  # Remove extra dimensions
        targets = targets.astype(np.uint8)  # Convert to uint8 type
    
    # Create image grid
    plt.figure(figsize=(15, 5))
    
    # Show original image
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(images)
    plt.axis('off')
    
    # Show target mask
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(targets, cmap='gray')
    plt.axis('off')
    
    # Show prediction result
    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(pred_binary, cmap='gray')
    plt.axis('off')
    
    # Save image
    plt.savefig(os.path.join(save_path, f'batch_{batch_idx}.png'))
    plt.close()

def get_optimizer(config, model):
    """Get optimizer"""
    if config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.opt}")

def get_scheduler(config, optimizer):
    """Get learning rate scheduler"""
    if config.sch == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    else:
        raise ValueError(f'Scheduler {config.sch} not supported')

def get_dataloader_config(config):
    """Get dataloader configuration adjusted for worker number"""
    return {
        'batch_size': config.batch_size,
        'shuffle': True,
        'pin_memory': config.pin_memory,
        'num_workers': config.num_workers,
        'persistent_workers': config.persistent_workers if config.num_workers > 0 else False,
        'prefetch_factor': config.prefetch_factor if config.num_workers > 0 else None
    }