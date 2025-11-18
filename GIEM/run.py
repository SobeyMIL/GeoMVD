import sys
import os
# 获取当前run.py所在目录（即GIEM目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接得到src目录的绝对路径
src_dir = os.path.join(current_dir, "src")
# 将src目录加入Python搜索路径
sys.path.append(src_dir)

from src.remove_bg import remove_background
from src.run_geowizard import geowizard_inference
from src.utils import filter_image_files
from src.run_bini import bini_inference
from src.run_pyvista import pyvista_inference
from tqdm import tqdm
import argparse


INPUT_PATH = './INPUT'  # 输入图像路径
OUTPUT_PATH = './OUTPUT1'  # 输出图像路径

parser = argparse.ArgumentParser(description="Run MonoDepthNormal Estimation using Stable Diffusion.")
parser.add_argument("--INPUT_PATH", type=str, default=INPUT_PATH)
parser.add_argument("--OUTPUT_PATH", type=str, default=OUTPUT_PATH)

parser.add_argument("--geowizard_ckpt_path", type=str, default='E:/CheckPoint/Geowizard', 
                    help="Geowizard checkpoint path")
parser.add_argument("--stable_diffusion2_ckpt_path", type=str, default='E:/CheckPoint/diffusion-safetensors/stable-diffusion-2', 
                    help="Stable Diffusion 2 checkpoint path")
parser.add_argument("--seed", type=int, default=3047, help="seed to set the random state")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
parser.add_argument("--denoise_steps", type=int, default=10, help="Diffusion denoising steps")
parser.add_argument("--ensemble_size", type=int, default=3, help="Number of predictions to be ensembled")
parser.add_argument("--color_map", type=str, default='Spectral', help="Colormap used to render depth predictions")

parser.add_argument('--use_K', action='store_true', help="是否使用相机内参")
parser.add_argument('--k', type=float, default=2, help="双边法线积分的参数k, 越小越平滑")
parser.add_argument('--iter', type=int, default=500, help="最大迭代次数")
parser.add_argument('--tol', type=float, default=1e-4, help="相对能量变化容忍度")
args = parser.parse_args()




if __name__ == "__main__":
    input_image_list = os.listdir(args.INPUT_PATH)
    input_image_list = filter_image_files(input_image_list)
    print(f"Detected {len(input_image_list)} images")
    
    for input_image in tqdm(input_image_list, desc="Removing background", leave=True):
        input_image_path = os.path.join(args.INPUT_PATH, input_image)
        
        input_image_name = input_image.split('.')[0]
        output_path = os.path.join(args.OUTPUT_PATH, input_image_name)
        os.makedirs(output_path, exist_ok=True)
        output_image_path = os.path.join(output_path, 'input.png')
        mask_output_path = os.path.join(output_path, 'mask.png')
        remove_background(input_image_path, output_image_path, mask_output_path=mask_output_path, resize=True)
    
    geowizard_inference(args)
    bini_inference(args)
    pyvista_inference(args)
    
    
    
    
    
    
    
        
    
    