"""
Masked Autoencoder (MAE) implementation.

As described in:
"He et al. (2022). Masked Autoencoders Are Scalable Vision Learners."

Key features:
- Asymmetric encoder-decoder architecture
- Encoder operates only on visible patches (no mask tokens)
- Lightweight decoder handles full set with mask tokens
- High masking ratio (75%) for challenging task
- Pixel-level reconstruction with MSE loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from einops import rearrange
from mae.models.vit import VisionTransformer, TransformerBlock


class MAEEncoder(VisionTransformer):
    """
    MAE Encoder - operates only on visible patches.
    
    Differences from standard ViT:
    - No class token (or optional)
    - Processes only unmasked patches
    - Returns latent representation for visible patches
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Override to ensure no class token for encoder
        # The class token can be added before fine-tuning
        self.use_cls_token = False
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional masking.
        
        Args:
            x: Input images [B, C, H, W] or patch embeddings [B, N, C]
            mask: Binary mask [B, N] where 1=keep, 0=mask
            
        Returns:
            Encoded features for visible patches
        """
        if mask is not None:
            # Apply mask to patch embeddings
            if x.ndim == 4:
                x = self.patch_embed(x)  # [B, N, C]
            
            # Keep only unmasked patches
            B, N, C = x.shape
            x_visible = x[torch.arange(B).unsqueeze(1), mask.bool()]  # [B, num_visible, C]
            
            # Add positional embeddings (only to visible patches)
            pos_embed_visible = self.pos_embed[:, 1:, :] if self.use_cls_token else self.pos_embed
            x_visible = x_visible + pos_embed_visible[mask.bool()]
            
            # Apply transformer blocks
            for blk in self.blocks:
                x_visible = blk(x_visible)
            
            return self.norm(x_visible)
        else:
            # No mask, process all patches (for fine-tuning)
            return super().forward(x)


class MAEDecoder(nn.Module):
    """
    MAE Decoder - lightweight decoder for reconstruction.
    
    Takes encoded visible patches and mask tokens, reconstructs the
    original image pixels.
    
    Architecture:
    - Input: encoded visible patches + mask tokens (with positional embeddings)
    - Transformers blocks (shallow and narrow)
    - Output: pixel values for masked patches
    """
    
    def __init__(self,
                 num_patches: int,
                 embed_dim: int = 768,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 decoder_mlp_ratio: float = 4.0,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 drop_path: float = 0.0):
        """
        Initialize MAE decoder.
        
        Args:
            num_patches: Total number of patches
            embed_dim: Encoder embedding dimension
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder transformer blocks
            decoder_num_heads: Number of attention heads in decoder
            decoder_mlp_ratio: MLP ratio in decoder
            patch_size: Patch size (for output dimension)
            in_chans: Number of input channels
            attn_drop: Attention dropout
            proj_drop: Projection dropout
            drop_path: Stochastic depth rate
        """
        super().__init__()
        
        self.num_patches = num_patches
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        # Projection from encoder to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token (shared learned vector)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Positional embeddings for all patches
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path, decoder_depth)]
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=decoder_mlp_ratio,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=dpr[i]
            )
            for i in range(decoder_depth)
        ])
        
        # Final LayerNorm
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Output projection to pixel values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, 
            patch_size ** 2 * in_chans,
            bias=True
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights."""
        import math
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, 
                x_encoded: torch.Tensor,
                mask: torch.Tensor,
                ids_keep: Optional[torch.Tensor] = None,
                ids_restore: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of decoder.
        
        Args:
            x_encoded: Encoded visible patches [B, num_visible, embed_dim]
            mask: Binary mask [B, N] where 1=keep, 0=mask
            ids_keep: Indices of kept patches [B, num_visible]
            ids_restore: Indices to restore original order [B, N]
            
        Returns:
            Reconstructed pixel values for all patches [B, N, patch_size^2 * C]
        """
        B = x_encoded.shape[0]
        N = self.num_patches
        num_visible = x_encoded.shape[1]
        
        # Project to decoder dimension
        x = self.decoder_embed(x_encoded)  # [B, num_visible, dec_embed_dim]
        
        # Expand mask tokens for masked patches
        mask_tokens = self.mask_token.repeat(B, N - num_visible, 1)
        
        # Concatenate visible patches and mask tokens
        # Use ids_restore if provided to restore original order
        if ids_restore is not None:
            x_full = torch.cat([x, mask_tokens], dim=1)  # [B, N, dec_embed_dim]
            x_full = torch.gather(x_full, dim=1, 
                                 index=ids_restore.unsqueeze(-1).repeat(1, 1, self.decoder_embed_dim))
        else:
            # Assume x is already in correct order
            x_full = torch.cat([x, mask_tokens], dim=1)
        
        # Add positional embeddings
        x_full = x_full + self.decoder_pos_embed
        
        # Apply transformer blocks
        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        
        # Final LayerNorm
        x_full = self.decoder_norm(x_full)
        
        # Predict pixel values for all patches
        x_pred = self.decoder_pred(x_full)  # [B, N, patch_size^2 * C]
        
        return x_pred


class MaskedAutoencoder(nn.Module):
    """
    Full Masked Autoencoder with asymmetric encoder-decoder.
    
    Architecture:
    1. Encoder: Processes only visible patches (25% of patches)
    2. Decoder: Lightweight network that handles full set with mask tokens
    
    Pre-training task: Reconstruct missing pixels for masked patches.
    """
    
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 encoder_embed_dim: int = 768,
                 encoder_depth: int = 12,
                 encoder_num_heads: int = 12,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 drop_path: float = 0.0,
                 norm_pix_loss: bool = False):
        """
        Initialize Masked Autoencoder.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_chans: Number of input channels
            encoder_embed_dim: Encoder embedding dimension
            encoder_depth: Number of encoder blocks
            encoder_num_heads: Number of encoder heads
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder blocks
            decoder_num_heads: Number of decoder heads
            mlp_ratio: MLP ratio
            attn_drop: Attention dropout
            proj_drop: Projection dropout
            drop_path: Stochastic depth rate
            norm_pix_loss: Normalize pixel targets per patch
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss
        
        # Patch embedding parameters
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size ** 2 * in_chans
        
        # Encoder (processes only visible patches)
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            use_cls_token=False
        )
        
        # Decoder (lightweight, processes full set)
        self.decoder = MAEDecoder(
            num_patches=self.num_patches,
            embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            decoder_mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            in_chans=in_chans,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MAE weights."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, 
                x: torch.Tensor,
                mask_ratio: float = 0.75,
                return_reconstruction: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MAE.
        
        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Ratio of patches to mask (default: 0.75)
            return_reconstruction: Return full reconstruction (for visualization)
            
        Returns:
            Dictionary containing:
                - loss: Reconstruction loss
                - pred: Predicted pixel values for masked patches
                - target: Target pixel values for masked patches
                - mask: Binary mask
                - reconstruction: Full reconstructed image (optional)
        """
        B = x.shape[0]
        
        # Get patch embeddings
        x_patch = self.encoder.patch_embed(x)  # [B, N, C]
        
        # Generate random mask
        mask, ids_keep, ids_restore = self._random_masking(x_patch, mask_ratio)
        
        # Encoder: process only visible patches
        # Get positional embeddings for visible patches
        pos_embed = self.encoder.pos_embed[:, 1:, :] if self.encoder.use_cls_token else self.encoder.pos_embed
        x_visible = x_patch + pos_embed
        x_visible = x_visible[torch.arange(B).unsqueeze(1), mask.bool()]  # [B, num_visible, C]
        
        # Pass through encoder blocks
        for blk in self.encoder.blocks:
            x_visible = blk(x_visible)
        x_encoded = self.encoder.norm(x_visible)  # [B, num_visible, C]
        
        # Decoder: process full set with mask tokens
        x_pred = self.decoder(x_encoded, mask, ids_keep, ids_restore)  # [B, N, patch_dim]
        
        # Compute loss only on masked patches
        loss = self._compute_loss(x_pred, x_patch, mask)
        
        # Prepare output
        output = {
            'loss': loss,
            'pred': x_pred,
            'mask': mask,
            'ids_keep': ids_keep,
            'ids_restore': ids_restore
        }
        
        if return_reconstruction:
            output['reconstruction'] = self._reconstruct_image(x_pred, x, mask, ids_restore)
        
        return output
    
    def _random_masking(self, 
                        x: torch.Tensor, 
                        mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random mask for patches.
        
        Args:
            x: Patch embeddings [B, N, C]
            mask_ratio: Ratio of patches to mask
            
        Returns:
            mask: Binary mask [B, N] (1=keep, 0=mask)
            ids_keep: Indices of kept patches [B, num_keep]
            ids_restore: Indices to restore original order [B, N]
        """
        B, N, C = x.shape
        num_keep = int(N * (1 - mask_ratio))
        
        # Generate random permutation for each batch
        noise = torch.rand(B, N, device=x.device)
        
        # Get indices sorted by noise
        ids_shuffle = torch.argsort(noise, dim=1)  # [B, N]
        ids_keep = ids_shuffle[:, :num_keep]       # [B, num_keep]
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # [B, N]
        
        # Create mask
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_keep, True)
        
        return mask, ids_keep, ids_restore
    
    def _compute_loss(self, 
                      pred: torch.Tensor, 
                      target: torch.Tensor, 
                      mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss only on masked patches.
        
        Args:
            pred: Predicted pixel values [B, N, patch_dim]
            target: Target pixel values [B, N, patch_dim]
            mask: Binary mask [B, N] (1=keep, 0=mask)
            
        Returns:
            MSE loss on masked patches
        """
        # Apply per-patch normalization if enabled
        if self.norm_pix_loss:
            target = self._normalize_patches(target)
        
        # Compute MSE loss
        loss = (pred - target) ** 2
        
        # Only compute loss on masked patches
        loss = loss.mean(dim=-1)  # [B, N]
        loss = loss[~mask].mean()
        
        return loss
    
    def _normalize_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Normalize pixel values per patch (mean=0, std=1).
        
        Args:
            patches: Patch pixel values [B, N, patch_dim]
            
        Returns:
            Normalized patches
        """
        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.std(dim=-1, keepdim=True) + 1e-6
        return (patches - mean) / std
    
    def _reconstruct_image(self,
                           pred: torch.Tensor,
                           x: torch.Tensor,
                           mask: torch.Tensor,
                           ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct full image from predictions.
        
        Args:
            pred: Predicted patches [B, N, patch_dim]
            x: Original image [B, C, H, W]
            mask: Binary mask [B, N]
            ids_restore: Indices to restore order [B, N]
            
        Returns:
            Reconstructed image [B, C, H, W]
        """
        B, N, _ = pred.shape
        H = W = int(N ** 0.5)
        
        # Combine predictions with original visible patches
        x_patch = self.encoder.patch_embed(x)  # [B, N, C]
        
        # Reshape predictions to patch dimensions
        pred_patches = pred.view(B, H, W, self.patch_size, self.patch_size, self.in_chans)
        pred_patches = pred_patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        pred_patches = pred_patches.view(B, self.in_chans, H * self.patch_size, W * self.patch_size)
        
        # For visible patches, use original image
        # This is a simplified reconstruction
        return pred_patches
    
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder forward pass (for fine-tuning).
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Encoded features
        """
        return self.encoder(x)
    
    @property
    def output_dim(self) -> int:
        """Return encoder output dimension."""
        return self.encoder.embed_dim