import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


from diffusers import DDIMScheduler
from torchvision.utils import save_image
from torchvision.io import read_image

from diffusion_for_icl import ICLPipeline
from attention_icl_utils import register_attention_control_ICL, SelfAttentionControlICL


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipe = ICLPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

def load_512_cpu(image_path, left=0, right=0, top=0, bottom=220, device=None):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = Image.fromarray(image).resize((512, 512))

    return image



def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image

self_replace_steps = .8
NUM_DIFFUSION_STEPS = 2

out_dir = "/Users/jannabruner/Documents/research/SL_repo/results"

source_img_path = "/Users/jannabruner/Documents/research/SL_repo/images/000000196185_512.jpg"
source_mask_path = "/Users/jannabruner/Documents/research/SL_repo/images/000000196185_mask_512.jpg"

source_image = load_image(source_img_path, device)
source_image_mask = load_image(source_mask_path, device)

# invert the source image
# the first element is z_0, noised in latent space
# second element is the z_t intermediate latents. 
# first element of the list is z_T (clean representation) and the last element is z_0 (noised representation)

start_code_a, latents_list_a = pipe.invert(source_image,
                                        "",
                                        guidance_scale=7.5,
                                        num_inference_steps=2,
                                        return_intermediates=True)

# start_code_b, latents_list_b = pipe.invert(source_image_mask,
#                                         "",
#                                         guidance_scale=7.5,
#                                         num_inference_steps=4,
#                                         return_intermediates=True)

start_code_b, latents_list_b  = start_code_a, latents_list_a
start_code_c, latents_list_c  = start_code_a, latents_list_a

ref_latents = {'k':latents_list_a,'q':latents_list_c, 'v':latents_list_b} 
prompts = ["", "","",""]

start_latents = start_code_c.expand(len(prompts), -1, -1, -1)

controller = SelfAttentionControlICL(NUM_DIFFUSION_STEPS, self_replace_steps=self_replace_steps)

register_attention_control_ICL(pipe, controller)

results = pipe(prompts,
                    latents=start_latents,
                    guidance_scale=3.5,
                    ref_intermediate_latents=ref_latents,
                    num_inference_steps=NUM_DIFFUSION_STEPS)

save_image(results, os.path.join(out_dir, '000000196185_512_pred.jpg'))