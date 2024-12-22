import pathlib
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from config import RunConfig
import torch 
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_images(cfg: RunConfig, save_path: Optional[pathlib.Path] = None) -> Tuple[Image.Image, Image.Image]:
    image_style = load_size(cfg.app_image_path)
    image_struct = load_size(cfg.struct_image_path)
    if save_path is not None:
        Image.fromarray(image_style).save(save_path / f"in_style.png")
        Image.fromarray(image_struct).save(save_path / f"in_struct.png")
    return image_style, image_struct


def load_size(image_path: pathlib.Path,
              left: int = 0,
              right: int = 0,
              top: int = 0,
              bottom: int = 0,
              size: int = 512) -> Image.Image:
    if isinstance(image_path, (str, pathlib.Path)):
        image = np.array(Image.open(str(image_path)).convert('RGB'))  
    else:
        image = image_path

    h, w, _ = image.shape

    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]

    h, w, c = image.shape

    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    image = np.array(Image.fromarray(image).resize((size, size)))
    return image

def resize_mask(image_mask_path: Path, res=(64,64), interpolation=cv2.INTER_NEAREST):
    """
    resize a given mask
    
    Args:
        image_mask (numpy.ndarray): The binary mask (e.g., 512x512).
        res (tuple): resolution of the new mask (h,w)
        interpolation (cv2 interpolation type): a method for interpolating 

    Returns:
        resized mask (e.g., 64x64).
    """

    image_mask = load_size(image_mask_path)

    h, w = res[0], res[1]
    
    # Resize mask using nearest-neighbor interpolation
    resized_mask = cv2.resize(image_mask, (w, h), interpolation=interpolation)
    resized_mask =  torch.from_numpy(np.array(resized_mask)).float() / 255.0
    # mask_64 = mask_64.permute(2, 0, 1).unsqueeze(0).to(device)

    # input_mask = torch.from_numpy(np.array(image_mask)).float() / 127.5 - 1.0

    # x0 = input_mask.permute(2, 0, 1).unsqueeze(0).to(device)
    # latent_mask = (sd_model.vae.encode(x0).latent_dist.mode() * 0.18215).float()

  
    # latents = (1 / 0.18215) * latents
    # with torch.no_grad():
    #     image = vae.decode(latents).sample
    # image = (image / 2 + 0.5).clamp(0, 1)

    return resized_mask[:,:,0]

def attention_per_head(query_activations):
    # Extract query and reshape for 8 heads
    query_tensor = query_activations[0].squeeze(0)  # Shape: [4096, 320]
    query_heads = query_tensor.view(4096, 8, 40)  # Shape: [4096, 8, 40]

    # Average over head dimensions (like before)
    query_head_1 = query_heads[:, 0, :].mean(dim=1).view(64, 64).numpy()  # For head 1
    query_head_2 = query_heads[:, 1, :].mean(dim=1).view(64, 64).numpy()  # For head 2

    # Plot the first two attention heads
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(query_head_1, cmap='viridis')
    ax[0].set_title('Query Head 1')

    ax[1].imshow(query_head_2, cmap='viridis')
    ax[1].set_title('Query Head 2')

    plt.show()

def attention_average_all_features(query_activations):
    import numpy as np

# Take the first layer's query (you might want to loop over query_activations)
    query_tensor = query_activations[0]  # Shape: [1, 4096, 320]
    query_tensor = query_tensor.squeeze(0)  # Remove batch dimension: [4096, 320]

    # Average over feature dimensions (optionally, you can take PCA here)
    query_spatial = query_tensor.mean(dim=1)  # Shape: [4096]
    query_spatial = query_spatial.view(64, 64).numpy()  # Reshape to (64, 64)

    # Plot the query as a heatmap
    plt.imshow(query_spatial, cmap='viridis')
    plt.colorbar()
    plt.title('Query Activation Heatmap')
    plt.show()

def attention_to_PCA(query_activations):
    from sklearn.decomposition import PCA

    query_tensor = query_activations[0].squeeze(0)  # Shape: [4096, 320]

    # Reduce dimensionality from 320 to 2
    pca = PCA(n_components=2)
    query_reduced = pca.fit_transform(query_tensor)  # Shape: [4096, 2]

    # Normalize and reshape each channel into (64, 64)
    channel_1 = query_reduced[:, 0].reshape(64, 64)
    channel_2 = query_reduced[:, 1].reshape(64, 64)

    # Plot the two PCA components
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(channel_1, cmap='viridis')
    ax[0].set_title('PCA Component 1')

    ax[1].imshow(channel_2, cmap='viridis')
    ax[1].set_title('PCA Component 2')

    plt.show()


def save_generated_masks(model, cfg: RunConfig):
    tensor2im(model.image_app_mask_32).save(cfg.output_path / f"mask_style_32.png")
    tensor2im(model.image_struct_mask_32).save(cfg.output_path / f"mask_struct_32.png")
    tensor2im(model.image_app_mask_64).save(cfg.output_path / f"mask_style_64.png")
    tensor2im(model.image_struct_mask_64).save(cfg.output_path / f"mask_struct_64.png")


def tensor2im(x) -> Image.Image:
    return Image.fromarray(x.cpu().numpy().astype(np.uint8) * 255)