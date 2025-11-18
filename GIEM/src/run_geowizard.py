import os
from lightning import seed_everything
import torch
import argparse
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from geowizard.models.geowizard_v2_pipeline import DepthNormalEstimationPipeline

import time
from tqdm import tqdmS
from PIL import Image
import numpy as np


INPUT_PATH = 'GIEM/INPUT'  # 输入图像路径
OUTPUT_PATH = 'GIEM/OUTPUT'  # 输出图像路径

geowizard_ckpt_path = 'E:/CheckPoint/Geowizard' 
# https://huggingface.co/lemonaddie/Geowizard/tree/main
stable_diffusion2_ckpt_path = 'E:/CheckPoint/diffusion-safetensors/stable-diffusion-2' 
# https://huggingface.co/stabilityai/stable-diffusion-2

device = "cuda:0"
denoise_steps = 10
ensemble_size = 3
color_map = 'Spectral'
seed = 0

parser = argparse.ArgumentParser(description="Run MonoDepthNormal Estimation using Stable Diffusion.")
parser.add_argument("--INPUT_PATH", type=str, default=INPUT_PATH)
parser.add_argument("--OUTPUT_PATH", type=str, default=OUTPUT_PATH)
parser.add_argument("--geowizard_ckpt_path", type=str, default=geowizard_ckpt_path, help="Geowizard checkpoint path")
parser.add_argument("--stable_diffusion2_ckpt_path", type=str, default=stable_diffusion2_ckpt_path, help="Stable Diffusion 2 checkpoint path")
parser.add_argument("--seed", type=int, default=seed, help="seed to set the random state")
parser.add_argument("--device", type=str, default=device, help="Device to use")
parser.add_argument("--denoise_steps", type=int, default=denoise_steps, help="Diffusion denoising steps")
parser.add_argument("--ensemble_size", type=int, default=ensemble_size, help="Number of predictions to be ensembled")
parser.add_argument("--color_map", type=str, default=color_map, help="Colormap used to render depth predictions")
args = parser.parse_args()



def geowizard_inference(args):
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    vae = AutoencoderKL.from_pretrained(args.stable_diffusion2_ckpt_path, subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.stable_diffusion2_ckpt_path, subfolder='text_encoder')
    scheduler = DDIMScheduler.from_pretrained(args.stable_diffusion2_ckpt_path, subfolder='scheduler')
    tokenizer = CLIPTokenizer.from_pretrained(args.stable_diffusion2_ckpt_path, subfolder='tokenizer')
    unet = UNet2DConditionModel.from_pretrained(args.geowizard_ckpt_path, subfolder='unet_v2')
               
    pipeline = DepthNormalEstimationPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler
    )
    pipeline = pipeline.to(device=device, dtype=dtype)
    
    input_name_list = os.listdir(args.OUTPUT_PATH)
    time_start = time.time()
    
    with torch.inference_mode():
        for input_name in tqdm(input_name_list, desc="Estimating Depth & Normal", leave=True):
            rgb_path = os.path.join(args.OUTPUT_PATH, input_name, "input.png")
            input_image = Image.open(rgb_path)

            pipeline_out = pipeline(
                input_image,
                denoising_steps = args.denoise_steps,
                ensemble_size= args.ensemble_size,
                processing_res = 768,
                match_input_res = True,
                domain = 'object',
                color_map = color_map,
                show_progress_bar = True,
            )
            depth_pred:     np.ndarray  = pipeline_out.depth_np
            depth_colored:  Image.Image = pipeline_out.depth_colored
            normal_pred:    np.ndarray  = pipeline_out.normal_np
            normal_colored: Image.Image = pipeline_out.normal_colored
            
            save_path = os.path.join(args.OUTPUT_PATH, input_name)
            geo_save_path = os.path.join(save_path, "GeoWirzed_output")
            os.makedirs(geo_save_path, exist_ok=True)
            # 保存几何信息
            npy_save_path = os.path.join(geo_save_path, "depth_pred.npy")
            np.save(npy_save_path, depth_pred)
            normal_npy_save_path = os.path.join(geo_save_path, "normal_pred.npy")
            np.save(normal_npy_save_path, normal_pred)
            # Colorize
            depth_colored_save_path = os.path.join(geo_save_path, "depth_colored.png")
            depth_colored.save(depth_colored_save_path)
            normal_colored_save_path = os.path.join(geo_save_path, "normal_colored.png")
            normal_colored.save(normal_colored_save_path)
            
            input_image.save(os.path.join(geo_save_path, "mask.png"))
    del pipeline
    torch.cuda.empty_cache()  # 清空缓存
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")


if __name__ == "__main__":

    geowizard_inference(args)
