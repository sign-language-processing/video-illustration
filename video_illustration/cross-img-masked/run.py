import sys
from typing import List

import numpy as np
import pyrallis
import torch
from PIL import Image
from diffusers.training_utils import set_seed
from typing import NamedTuple, Optional
from pathlib import Path

sys.path.append(".")
sys.path.append("..")

from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig, Range
from utils import latent_utils
from utils.latent_utils import load_latents_or_invert_images

class Range(NamedTuple):
    start: int
    end: int


device = 'cuda' if torch.cuda.is_available() else "cpu"
@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)


def run(cfg: RunConfig) -> List[Image.Image]:
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))
    set_seed(cfg.seed)
    model = AppearanceTransferModel(cfg)
    latents_app, latents_struct, noise_app, noise_struct = load_latents_or_invert_images(model=model, cfg=cfg)

    model.set_attention_masks(cfg.query_mask_path, cfg.key_mask_path, cfg.value_mask_path)
    model.set_latents(latents_app, latents_struct)
    model.set_noise(noise_app, noise_struct)
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
        prompt=[cfg.prompt] * 3,
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
    images[0].save(cfg.output_path / f"out_transfer---seed_{cfg.seed}.png")
    images[1].save(cfg.output_path / f"out_style---seed_{cfg.seed}.png")
    images[2].save(cfg.output_path / f"out_struct---seed_{cfg.seed}.png")
    joined_images = np.concatenate(images[::-1], axis=1)
    Image.fromarray(joined_images).save(cfg.output_path / f"out_joined---seed_{cfg.seed}.png")
    return images


if __name__ == '__main__':
    main()
    # cfg = RunConfig(
    #     app_image_path = Path("/Users/jannabruner/Documents/research/SL_repo_ongoing/SL_repo/images/fangen.jpg"),
    #     struct_image_path = Path("/Users/jannabruner/Documents/research/SL_repo_ongoing/SL_repo/images/bonn_frame_0026.jpg"),
    #     query_mask_path=Path("/Users/jannabruner/Documents/research/SL_repo_ongoing/SL_repo/images/left_hand_mask.png"),
    #     value_mask_path=Path("/Users/jannabruner/Documents/research/SL_repo_ongoing/SL_repo/images/left_hand_mask.png"),
    #     use_masked_adain=False,
    #     seed=42,
    #     domain_name="a woman",
    #     prompt=None,
    #     load_latents=True,
    #     skip_steps=0,
    #     num_timesteps=1,
    #     cross_attn_32_range=Range(start=10,end=10),
    #     cross_attn_64_range=Range(start=0,end=2),
    #     adain_range=Range(start=25,end=45),
    #     swap_guidance_scale=3.5,
    #     contrast_strength=1.67,
    #     query_scale = 3.5,
    #     value_scale=3.5
    # )

    # run(cfg)
