import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision.utils import make_grid
import torchvision.transforms as T  # Fixed import statement

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(
        input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps,
        ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn,
        num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout,
        conditional=args.use_cfg, c_dim=args.unet_ch
    )
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # Create scheduler
    scheduler_args = dict(
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        variance_type=args.variance_type,
        prediction_type=args.prediction_type,
        clip_sample=args.clip_sample,
        clip_sample_range=args.clip_sample_range
    )
    scheduler = DDPMScheduler(**scheduler_args)
    
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    
    # cfg
    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(embed_dim=args.unet_ch, n_classes=args.num_classes)
        
    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler
    if args.use_ddim:
        scheduler_class = DDIMScheduler
    else:
        scheduler_class = DDPMScheduler
    # Create scheduler
    scheduler = scheduler_class(**scheduler_args)

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # Create pipeline
    pipeline = DDPMPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        class_embedder=class_embedder
    )

    logger.info("***** Running Inference *****")
    
    # Run inference to generate images
    all_images = []
    batch_size = 50 if args.use_cfg else 100
    
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(batch_size=batch_size, classes=classes, guidance_scale=args.cfg_guidance_scale, generator=generator)
            # gen_images: list of PIL images, convert to tensor
            gen_images_tensor = torch.stack([T.ToTensor()(img) for img in gen_images])
            all_images.append(gen_images_tensor)
        all_images = torch.cat(all_images, dim=0)
    else:
        # generate 5000 images
        total = 1000
        gen_so_far = 0
        for _ in tqdm(range(0, total, batch_size)):
            current_batch = min(batch_size, total - gen_so_far)
            gen_images = pipeline(batch_size=current_batch, generator=generator)
            gen_images_tensor = torch.stack([T.ToTensor()(img) for img in gen_images])
            all_images.append(gen_images_tensor)
            gen_so_far += current_batch
        all_images = torch.cat(all_images, dim=0)
    
    # Load validation images as reference batch
    from torchvision import datasets
    val_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    
    val_data_dir = args.data_dir.replace('train', 'val') if 'train' in args.data_dir else args.data_dir
    val_dataset = datasets.ImageFolder(val_data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)
    ref_images = []
    for images, _ in val_loader:
        ref_images.append(images)
        if len(ref_images) * 100 >= 1000:
            break
    ref_images = torch.cat(ref_images, dim=0)[:1000]
    
    # Using torchmetrics for evaluation
    import torchmetrics 
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    
    # Compute FID and IS
    # Note: FID expects images in range [0, 255] as uint8
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    iscore = InceptionScore(normalize=True).to(device)
    
    # Process images to uint8 range for FID
    all_images_uint8 = (all_images * 255).to(torch.uint8)
    ref_images_uint8 = ((ref_images * 0.5 + 0.5) * 255).to(torch.uint8)  # Denormalize reference images
    
    # FID
    fid.update(all_images_uint8.to(device), real=False)
    fid.update(ref_images_uint8.to(device), real=True)
    fid_score = fid.compute().item()
    
    # IS
    iscore.update(all_images_uint8.to(device))
    is_mean, is_std = iscore.compute()
    
    # Log results
    logger.info(f"FID: {fid_score:.4f}, IS: {is_mean:.4f} ± {is_std:.4f}")
    print(f"FID: {fid_score:.4f}, IS: {is_mean:.4f} ± {is_std:.4f}")
    
    # Save results to file
    results = {
        "fid": fid_score,
        "is_mean": is_mean.item(),
        "is_std": is_std.item(),
        "model": args.ckpt,
        "use_cfg": args.use_cfg,
        "guidance_scale": args.cfg_guidance_scale if args.use_cfg else None,
        "use_ddim": args.use_ddim,
        "num_inference_steps": args.num_inference_steps,
        "latent_ddpm": args.latent_ddpm
    }
    
    output_dir = os.path.dirname(args.ckpt)
    import json
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()