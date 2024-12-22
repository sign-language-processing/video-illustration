# Cross Image Attention with Masks 

Implementation copied from https://github.com/garibida/cross-image-attention. 

Original code with multiplying a masked area (such as hands) on key/query/value (before softmax) by factor scale  

additional config params added:
query_mask_path
query_scale
key_mask_path
key_scale
value_mask_path
value_scale
inject_64_res
inject_32_res

for best results in the config file: 
1. disable the injection of 32x32 attention layers of the decoder (only 64x64 attention layers of the decoder will be modified)
2. use "domain_name". 
3. for this code set use_masked_adain=False (was not tested with masked adain) 

Note: "inject_32_res" param will be used only if a range set to "cross_attn_32_range" param. 



Setup and usage are the same as the above repository.


Use "app_image_path" as the reference style and "struct_image_path" as the driving image

## Example injecting a mask to query 

python run.py --app_image_path ../../images/reference_styles/32.png --struct_image_path ../../images/bonn_frame_0026/bonn_frame_0026.jpg --query_mask_path ../../images/bonn_frame_0026/left_hand_mask.png --query_scale 3.5 --output_path ../../results/ --use_masked_adain False --load_latents False --num_timesteps 100 --skip_steps 30 --contrast_strength 1.67 --swap_guidance_scale 3.5 --domain_name "a woman" 
