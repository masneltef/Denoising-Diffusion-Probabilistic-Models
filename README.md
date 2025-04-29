# Denoising Diffusion Probabilistic Models Implementation

This repository contains a comprehensive implementation of Denoising Diffusion Probabilistic Models (DDPMs) and their variants for high-quality image generation.

## Overview

This project implements several key diffusion model architectures and techniques:

- Standard DDPM (Denoising Diffusion Probabilistic Models)
- DDIM (Denoising Diffusion Implicit Models) for faster sampling
- Latent Diffusion Models (LDM) using VAE for computational efficiency
- Classifier-Free Guidance (CFG) for conditional generation
- Advanced techniques including improved beta scheduling and learned variance

## Dataset

The models were trained and evaluated on the ImageNet-100 dataset, containing 100 classes with approximately 130,000 training images at 128×128 resolution.

## Project Structure

```
├── configs/
│   └── ddpm.yaml             # Configuration file for DDPM training
├── models/
│   ├── unet.py               # U-Net architecture with timestep embeddings
│   ├── unet_modules.py       # ResBlocks, attention modules for U-Net
│   ├── vae.py                # Variational Autoencoder for latent space
│   └── class_embedder.py     # Class embedder for conditional generation
├── schedulers/
│   ├── scheduling_ddpm.py    # DDPM noise scheduler
│   └── scheduling_ddim.py    # DDIM noise scheduler
├── pipelines/
│   └── ddpm.py               # Pipeline for training and inference
├── utils/
│   └── utils.py              # Helper functions and utilities
├── train.py                  # Training script
├── eval.py                   # Evaluation script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+
- torchvision
- numpy
- wandb

Install dependencies:

```bash
pip install -r requirements.txt
```

## Pretrained Models

Pretrained VAE model is required for latent space diffusion:

```bash
mkdir -p pretrained
# Download VAE model
# Place at pretrained/model.ckpt
```

## Training

### Standard DDPM

```bash
python train.py --config configs/ddpm.yaml
```

### Latent DDPM

```bash
python train.py --config configs/ddpm.yaml --latent_ddpm True
```

### Conditional Generation with CFG

```bash
python train.py --config configs/ddpm.yaml --latent_ddpm True --use_cfg True
```

## Inference

```bash
python eval.py --ckpt experiments/exp-1/checkpoints/checkpoint_epoch_49.pth --use_ddim True --num_inference_steps 100 --use_cfg True --cfg_guidance_scale 2.0
```

## Results

| Model | FID ↓ | IS ↑ |
|-------|-------|------|
| DDPM (1000 steps) | 42.18 | 5.26 ± 0.31 |
| Latent DDPM | 36.42 | 7.18 ± 0.42 |
| Latent DDPM + CFG (w=2.0) | 28.76 | 9.32 ± 0.62 |
| Advanced Techniques | 21.12 | 12.24 ± 0.82 |

## Visualizations

Sample generated images and training curves can be found in the `figures/` directory.

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33, 6840-6851.
2. Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502.
3. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10684-10695).
