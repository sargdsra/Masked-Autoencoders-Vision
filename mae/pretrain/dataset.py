"""
Data loading and masking for MAE pre-training.

Handles:
- Image loading and augmentation (minimal augmentation)
- Patch extraction
- Mask generation
- Batch preparation
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import random
from typing import Tuple, Optional, List, Dict, Any


class MAEPretrainDataset(Dataset):
    """
    Dataset for MAE pre-training.
    
    Uses minimal augmentation (only random resized crop and horizontal flip)
    as shown in Table 1e of the paper.
    """
    
    def __init__(self,
                 image_paths: List[str],
                 img_size: int = 224,
                 patch_size: int = 16,
                 mask_ratio: float = 0.75,
                 use_augmentation: bool = True,
                 transform: Optional[callable] = None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            img_size: Input image size
            patch_size: Patch size
            mask_ratio: Ratio of patches to mask
            use_augmentation: Use random resized crop and flip
            transform: Optional custom transform
        """
        self.image_paths = image_paths
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.use_augmentation = use_augmentation
        
        # Number of patches per image
        self.num_patches = (img_size // patch_size) ** 2
        
        # Set transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._build_transform()
    
    def _build_transform(self) -> T.Compose:
        """Build default transform for MAE pre-training."""
        if self.use_augmentation:
            return T.Compose([
                T.RandomResizedCrop(self.img_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(self.img_size),
                T.CenterCrop(self.img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - image: Image tensor [C, H, W]
                - mask: Binary mask [N] (1=keep, 0=mask)
                - ids_keep: Indices of kept patches [num_keep]
                - ids_restore: Indices to restore order [N]
                - image_path: Original image path
        """
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform
        img_tensor = self.transform(img)
        
        # Generate mask
        mask, ids_keep, ids_restore = self._generate_mask()
        
        return {
            'image': img_tensor,
            'mask': mask,
            'ids_keep': ids_keep,
            'ids_restore': ids_restore,
            'image_path': img_path
        }
    
    def _generate_mask(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random mask for patches.
        
        Returns:
            mask: Binary mask [N] (1=keep, 0=mask)
            ids_keep: Indices of kept patches [num_keep]
            ids_restore: Indices to restore order [N]
        """
        N = self.num_patches
        num_keep = int(N * (1 - self.mask_ratio))
        
        # Random permutation
        noise = torch.rand(N)
        ids_shuffle = torch.argsort(noise)
        ids_keep = ids_shuffle[:num_keep]
        ids_restore = torch.argsort(ids_shuffle)
        
        # Create mask
        mask = torch.zeros(N, dtype=torch.bool)
        mask[ids_keep] = True
        
        return mask, ids_keep, ids_restore


class MAEPretrainDataLoader:
    """
    Data loader for MAE pre-training with efficient batch preparation.
    """
    
    @staticmethod
    def prepare_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare batch for MAE training.
        
        Args:
            batch: Batch from DataLoader
            
        Returns:
            Prepared batch with:
                - images: Image tensor [B, C, H, W]
                - mask: Binary mask [B, N]
                - ids_keep: Indices of kept patches [B, num_keep]
                - ids_restore: Indices to restore order [B, N]
        """
        return {
            'images': batch['image'],
            'mask': batch['mask'],
            'ids_keep': batch['ids_keep'],
            'ids_restore': batch['ids_restore']
        }
    
    @staticmethod
    def create_dataloader(dataset: Dataset,
                         batch_size: int = 256,
                         num_workers: int = 8,
                         shuffle: bool = True) -> DataLoader:
        """Create DataLoader for pre-training."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )