from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion import invert

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_latents_or_invert_images(model: AppearanceTransferModel, cfg: RunConfig):
    if cfg.load_latents and cfg.prompt_gt_image_path.exists() and cfg.prompt_image_path.exists() and cfg.query_image_path.exists():
        print("Loading existing latents...")
        latents_q, latents_v, latents_k = load_latents(cfg.prompt_gt_image_path, cfg.prompt_image_path, cfg.query_image_path)
        noise_q, noise_v, noise_k = load_noise(cfg.prompt_gt_image_path, cfg.prompt_image_path, cfg.query_image_path)
        print("Done.")
    else:
        print("Inverting images...")

        image_key, image_query, image_value = image_utils.load_images(cfg=cfg, save_path=cfg.output_path)
        model.enable_edit = False  # Deactivate the cross-image attention layers
        latents_q, latents_v, latents_k , noise_q, noise_v, noise_k  = invert_images(image_key=image_key,
                                                                             image_query=image_query,
                                                                             image_value=image_value,
                                                                             sd_model=model.pipe,
                                                                             cfg=cfg)
        model.enable_edit = True
        print("Done.")
    return latents_q, latents_v, latents_k , noise_q, noise_v, noise_k

def load_latents(prompt_gt_image_path: Path, prompt_image_path: Path, query_image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_k = torch.load(prompt_image_path)
    latents_v = torch.load(prompt_gt_image_path)
    latents_q = torch.load(query_image_path)
    if type(latents_k) == list:
        latents_k = [l.to(device) for l in latents_k]
        latents_v = [l.to(device) for l in latents_v]
        latents_q = [l.to(device) for l in latents_q]
    else:
        latents_k = latents_k.to(device)
        latents_v = latents_v.to(device)
        latents_q = latents_q.to(device)
    return latents_q, latents_v, latents_k 

def load_noise(prompt_gt_image_path: Path, prompt_image_path: Path, query_image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    noise_k = torch.load(prompt_image_path.parent / (prompt_image_path.stem + "_ddpm_noise.pt"))
    noise_q = torch.load(query_image_path.parent / (query_image_path.stem + "_ddpm_noise.pt"))
    noise_v = torch.load(prompt_gt_image_path.parent / (prompt_gt_image_path.stem + "_ddpm_noise.pt"))

    noise_k = noise_k.to(device)
    noise_q = noise_q.to(device)
    noise_v = noise_v.to(device)
    return  noise_q, noise_v, noise_k 

def invert_images(sd_model: AppearanceTransferModel, image_key: Image.Image, image_query: Image.Image, image_value:Image.Image, cfg: RunConfig):
    input_q = torch.from_numpy(np.array(image_query)).float() / 127.5 - 1.0
    input_k = torch.from_numpy(np.array(image_key)).float() / 127.5 - 1.0
    input_v = torch.from_numpy(np.array(image_value)).float() / 127.5 - 1.0


    zs_q, latents_q = invert(x0=input_q.permute(2, 0, 1).unsqueeze(0).to(device),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=3.5)
    
    zs_k, latents_k = invert(x0=input_k.permute(2, 0, 1).unsqueeze(0).to(device),
                                       pipe=sd_model,
                                       prompt_src=cfg.prompt,
                                       num_diffusion_steps=cfg.num_timesteps,
                                       cfg_scale_src=3.5)
    
    zs_v, latents_v = invert(x0=input_v.permute(2, 0, 1).unsqueeze(0).to(device),
                                       pipe=sd_model,
                                       prompt_src=cfg.prompt,
                                       num_diffusion_steps=cfg.num_timesteps,
                                       cfg_scale_src=3.5)
    # Save the inverted latents and noises
    torch.save(latents_q, cfg.latents_path / f"{cfg.query_image_path.stem}.pt")
    torch.save(latents_k, cfg.latents_path / f"{cfg.prompt_image_path.stem}.pt")
    torch.save(latents_v, cfg.latents_path / f"{cfg.prompt_gt_image_path.stem}.pt")

    torch.save(zs_q, cfg.latents_path / f"{cfg.query_image_path.stem}_ddpm_noise.pt")
    torch.save(zs_v, cfg.latents_path / f"{cfg.prompt_gt_image_path.stem}_ddpm_noise.pt")
    torch.save(zs_k, cfg.latents_path / f"{cfg.prompt_image_path.stem}_ddpm_noise.pt")

    return latents_q, latents_v, latents_k , zs_q, zs_v, zs_k 


def get_init_latents_and_noises(model: AppearanceTransferModel, cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    if model.latents_query.dim() == 4 and model.latents_key.dim() == 4 and model.latents_value.dim() == 4 and model.latents_key.shape[0] > 1:
        model.latents_query = model.latents_query[cfg.skip_steps]
        model.latents_key = model.latents_key[cfg.skip_steps]
        model.latents_value = model.latents_value[cfg.skip_steps]

    init_latents = torch.stack([model.latents_query, model.latents_query, model.latents_key, model.latents_value])
    init_zs = [model.zs_query[cfg.skip_steps:], model.zs_query[cfg.skip_steps:], model.zs_key[cfg.skip_steps:], model.zs_value[cfg.skip_steps:]]
    return init_latents, init_zs

