# Masked Autoencoders Are Scalable Vision Learners (MAE)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## 📖 About

PyTorch implementation of the paper:

**"Masked Autoencoders Are Scalable Vision Learners"**  
by Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick (2022)

MAE is a simple and scalable self-supervised learning method for Vision Transformers that:
- Masks random patches of input images (75%)
- Reconstructs missing pixels using an asymmetric encoder-decoder
- Achieves state-of-the-art results on ImageNet (87.8%)
- Scales effectively to very large models (ViT-Huge)

## 🎯 Key Features

- **Asymmetric Architecture**: Encoder processes only visible patches, lightweight decoder handles full set
- **High Masking Ratio**: 75% masking creates challenging pretext task
- **Minimal Augmentation**: No color jittering needed (unlike contrastive learning)
- **Efficient Training**: 3-4x speedup over using mask tokens in encoder
- **Scalable**: Works with ViT-Base, Large, and Huge models

## 🔧 Installation

```bash
git clone https://github.com/sargdsra/Masked-Autoencoders-Vision.git
cd Masked-Autoencoders-Vision
pip install -r requirements.txt
pip install -e .
```

## 🚀 Quick Start
```python
from mae.models.mae import MaskedAutoencoder
from mae.pretrain.dataset import MAEPretrainDataset, MAEPretrainDataLoader
from mae.pretrain.trainer import MAEPretrainTrainer

# Create model
model = MaskedAutoencoder(
    img_size=224,
    patch_size=16,
    encoder_embed_dim=768,   # ViT-Base
    encoder_depth=12,
    encoder_num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    norm_pix_loss=False
)

# Prepare dataset
dataset = MAEPretrainDataset(
    image_paths=image_paths,
    img_size=224,
    patch_size=16,
    mask_ratio=0.75
)

# Create trainer and train
trainer = MAEPretrainTrainer(model, dataloader, config)
results = trainer.train()
```

## Fine-tuning
```python
# Load pre-trained encoder
encoder = model.encoder

# Add classifier head
encoder.use_cls_token = True
classifier = nn.Linear(768, 1000)
model = nn.Sequential(encoder, classifier)

# Fine-tune on ImageNet
```

## 📚 References

- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR.

- Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.

- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.