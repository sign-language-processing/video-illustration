import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed

sys.path.append(".")
sys.path.append("..")

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images

device = 'cuda' if torch.cuda.is_available() else "cpu"
@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)


def run(cfg: RunConfig) -> List[Image.Image]:
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)
    model = AppearanceTransferModel(cfg)
    latents_q, latents_v, latents_k , noise_q, noise_v, noise_k = load_latents_or_invert_images(model=model, cfg=cfg)
    model.set_latents(latents_q,latents_k,latents_v)
    model.set_noise(noise_q, noise_k, noise_v)
    print("Running appearance transfer...")
    images = run_appearance_transfer(model=model, cfg=cfg)
    print("Done.")
    return images

def run_appearance_transfer(model: AppearanceTransferModel, cfg: RunConfig) -> List[Image.Image]:
    init_latents, init_zs = latent_utils.get_init_latents_and_noises(model=model, cfg=cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Activate our cross-image attention layers
    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
    images = model.pipe(
        prompt=[cfg.prompt] *4,
        latents=init_latents,
        guidance_scale=1.0,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator(device).manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step),
    ).images
    # Save images
    images[0].save(cfg.output_path / f"im_output---seed_{cfg.seed}.png")
    images[1].save(cfg.output_path / f"im_query---seed_{cfg.seed}.png")
    images[2].save(cfg.output_path / f"im_key_img_prompt---seed_{cfg.seed}.png")
    images[3].save(cfg.output_path / f"im_value_task_prompt---seed_{cfg.seed}.png")
    joined_images = np.concatenate(images, axis=1)
    Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}.png")
    return images


if __name__ == '__main__':
    main()
