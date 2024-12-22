# Cross Image ICL 

Implementation copied from https://github.com/garibida/cross-image-attention. 

In this code, different images are used as key,query,value instead of style and struct images. 

for query set query_image_path
for key set prompt_image_path
for value set prompt_gt_image_path

for this code set use_masked_adain=False (was not tested with masked adain) 

Setup and usage are the same as the above repository.

## Example
python run.py --prompt_gt_image_path ../../images/airplane/000000196185_mask_512.jpg --prompt_image_path ../../images/airplane/000000196185_512.jpg --query_image_path ../../images/airplane/000000196185_512.jpg --output_path ../../results/icl_recover_results_with_cross_img_basecode  --use_masked_adain False --load_latents False --num_timesteps 70 --skip_steps 0 --contrast_strength 1.67 --swap_guidance_scale 3.5
