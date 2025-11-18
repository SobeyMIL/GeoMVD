import os
import shutil
from unittest import result

input_path = r"E:\Dataset\GSO_PIPELINE_INPUT"

tag_list = [
    'input.png',
    'GT0.png',
    'output_lora_10000_stp50.png',
    'output_lora_10000_wo_dynamic_geo_scale0.32.png', 
    'output_lora_10000_wo_Geo_attention.png',
    'output_lora_10000_wo_Geo_hs_mask.png',
    'output_lora_10000_wo_global_condition.png',
    'output_lora_10000_wo_image_ref_atten.png',
    'zero123++result.png',
    'output_mv0.32_cfg3.0_stp50.png',
]

result_path = r"E:\Dataset\GSO_PIPELINE_INPUT_compare"
os.makedirs(result_path)

for image_name in os.listdir(input_path):
    image_name_path = os.path.join(input_path, image_name, "AAA_compare")
    to_path = os.path.join(result_path, image_name)
    os.makedirs(to_path, exist_ok=True)
    shutil.move(image_name_path, to_path)
    
    
    
    