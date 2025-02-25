from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torch
import random
import cv2

class NPY_datasets(Dataset):
    """Dataset loading class, supports image segmentation tasks for MSCB_UNet"""
    def __init__(self, path_Data, config, train=True):
        """
        Initialize the dataset
        Args:
            path_Data: Root directory of the dataset
            config: Configuration object
            train: Whether it is a training set
        """
        super(NPY_datasets, self).__init__()
        
        self.path_Data = path_Data
        self.train = train
        self.config = config
        
        # Validate data directory
        self._validate_directories()
        
        # Pre-validate all images
        self.valid_pairs = self._get_valid_pairs()
        print(f"Found {len(self.valid_pairs)} valid image pairs")

    def _validate_directories(self):
        """Validate the data directory structure"""
        if self.train:
            self.img_path = os.path.join(self.path_Data, 'train/images/')
            self.mask_path = os.path.join(self.path_Data, 'train/masks/')
        else:
            self.img_path = os.path.join(self.path_Data, 'val/images/')
            self.mask_path = os.path.join(self.path_Data, 'val/masks/')
            
        if not os.path.exists(self.img_path) or not os.path.exists(self.mask_path):
            raise RuntimeError(f"Data directories not found: {self.img_path} or {self.mask_path}")

    def _get_valid_pairs(self):
        """Pre-validate all image pairs"""
        valid_pairs = []
        images = sorted([f for f in os.listdir(self.img_path) 
                       if f.endswith(('.png', '.jpg', '.jpeg'))])
        masks = sorted([f for f in os.listdir(self.mask_path) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for img_name, mask_name in zip(images, masks):
            img_path = os.path.join(self.img_path, img_name)
            mask_path = os.path.join(self.mask_path, mask_name)
            
            try:
                # Validate if the image can be loaded correctly
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(mask_path) as mask:
                    mask.verify()
                valid_pairs.append((img_path, mask_path))
            except Exception as e:
                print(f"Skipping invalid pair ({img_name}, {mask_name}): {str(e)}")
                continue
                
        return valid_pairs

    def __getitem__(self, index):
        """Get data item without using recursion"""
        if not 0 <= index < len(self.valid_pairs):
            raise IndexError(f"Index {index} out of range")
            
        img_path, mask_path = self.valid_pairs[index]
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Convert to tensor
            img = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
            mask = torch.FloatTensor(np.array(mask)).unsqueeze(0) / 255.0
            
            return img, mask
            
        except Exception as e:
            # Return a default value in case of an error, instead of recursion
            print(f"Error loading data at index {index}: {str(e)}")
            # Return zero tensor as a substitute
            return torch.zeros((3, 256, 256)), torch.zeros((1, 256, 256))
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def _load_image(self, path):
        """Load image file"""
        try:
            # Use PIL to load the image
            img = Image.open(path).convert('RGB')
            img = np.array(img)
            
            # Normalize to [0,1] range
            if img.max() > 1:
                img = img / 255.0
                
            return img
            
        except Exception as e:
            print(f"Error loading image {path}: {str(e)}")
            raise
    
    def _load_mask(self, path):
        """Load mask file"""
        try:
            # Use PIL to load the mask
            mask = Image.open(path).convert('L')
            mask = np.array(mask)
            
            # Convert mask to binary image
            mask = (mask > 127).astype(np.float32)
            
            # Add channel dimension
            mask = np.expand_dims(mask, axis=2)
            
            return mask
            
        except Exception as e:
            print(f"Error loading mask {path}: {str(e)}")
            raise

def get_dataloader(config, train=True):
    """Get data loader, optimized for MSCB_UNet"""
    # Create dataset
    dataset = NPY_datasets(config.data_path, config, train)
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size if train else 1,
        shuffle=train,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=train,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    
    return loader

def get_mean_std(dataset):
    """Calculate the mean and standard deviation of the dataset"""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    
    return mean, std

class AugmentedDataset(Dataset):
    """Dataset class with additional data augmentation"""
    def __init__(self, base_dataset, augment_factor=2):
        """
        Initialize augmented dataset
        Args:
            base_dataset: Base dataset
            augment_factor: Augmentation factor
        """
        self.base_dataset = base_dataset
        self.augment_factor = augment_factor
        
    def __getitem__(self, index):
        """Get augmented data sample"""
        # Calculate the index of the original dataset
        base_idx = index // self.augment_factor
        aug_idx = index % self.augment_factor
        
        # Get original data
        img, mask = self.base_dataset[base_idx]
        
        # Apply different augmentations based on aug_idx
        if aug_idx > 0:
            img, mask = self._apply_extra_augmentation(img, mask, aug_idx)
            
        return img, mask
    
    def __len__(self):
        """Return the size of the augmented dataset"""
        return len(self.base_dataset) * self.augment_factor
    
    def _apply_extra_augmentation(self, img, mask, aug_type):
        """Apply additional data augmentation"""
        if aug_type == 1:
            # Add Gaussian noise
            noise = torch.randn_like(img) * 0.1
            img = img + noise
            img = torch.clamp(img, 0, 1)
        
        return img, mask