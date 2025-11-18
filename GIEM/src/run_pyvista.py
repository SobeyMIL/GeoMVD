import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True
import os
import torch
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from typing import List
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyvista.plotting.renderer")

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

# [30, -20, 30, -20, 30, -20]
def render_ply(
    ply_file_path: str,   # .ply
    save_path_mv: str,    # folder
    save_path_grid: str,  # folder
    azimuth: List[float]   = [30, 90, 150, 210, 270, 330], 
    elevation: List[float] = [ 0,  0,   0,   0,   0,   0], 
    resolution: int = 640,
):
    # 创建输出文件夹
    if not os.path.exists(save_path_mv):
        os.makedirs(save_path_mv, exist_ok=True)
    if not os.path.exists(save_path_grid):
        os.makedirs(save_path_grid, exist_ok=True)
    
    # 读取 .ply, 修正视角
    mesh = pv.read(ply_file_path)
    mesh.rotate_x(45, inplace=True)  
    mesh.rotate_y(-90, inplace=True) 
    
    # --------------------------------------------------------------------------------------------    
    normals = mesh['Normals']
    normals_norm = (normals + 1) / 2
    mesh['NormalsColor'] = (normals_norm * 255).astype(np.uint8)
    # --------------------------------------------------------------------------------------------
    
   
    for idx, (az, el) in enumerate(zip(azimuth, elevation)):
        plotter = pv.Plotter(
            off_screen=True,
            window_size=(resolution, resolution), 
        )
        plotter.set_background('white')  # 使用 set_background 设置背景颜色
        plotter.add_mesh(
            mesh, 
            style='surface', 
            scalars='NormalsColor', 
            edge_color="black", 
            show_edges=False, 
            show_scalar_bar=False,
            cmap='jet'
        )
        """
        常见的颜色映射列表
            viridis - 色调从深紫到亮黄，适用于视觉感知。
            plasma - 色调从紫色到黄色，颜色对比度强，适合高对比度数据。
            inferno - 色调从黑色到明亮黄色，适用于需要强调低值和高值的场景。
            magma - 色调从黑色到亮白色，适合传达较暗的场景。
            cividis - 色调从蓝色到黄色，优化了色盲用户的视觉体验。
            twilight - 颜色从蓝色到橙色，适用于温和的渐变显示。
            twilight_shifted - 颜色从蓝色到橙色，但色调有一定的偏移。
            RdBu - 红蓝色渐变，常用于表示正负数据或温度数据。
            coolwarm - 从蓝色到红色，通常用于显示温度数据。
            jet - 从蓝色到红色的经典颜色映射，尽管它常被认为有些不适合科学数据。
        """
        plotter.enable_anti_aliasing('fxaa')
        # 方位角
        plotter.camera.azimuth = az
        # 仰角 (修正)
        plotter.camera.elevation = el - 40
        plotter.show(screenshot=os.path.join(save_path_mv, f"{str(idx).zfill(3)}_azi{az}_el{el}.png"))
        
    mv_img_list = os.listdir(save_path_mv)
    mv_img_list = sorted(mv_img_list)
    
    IMG_list = [Image.open(os.path.join(save_path_mv, mv_img)).convert("RGB") for mv_img in mv_img_list]
    np_list = [np.asarray(img, dtype=np.float32) for img in IMG_list]
    np_list = [np.ascontiguousarray(np_arr).astype(np.float32) for np_arr in np_list]
    tr_list = [torch.tensor(np_arr, dtype=torch.float32).permute(2, 0, 1) for np_arr in np_list]


    batch_img = torch.stack(tr_list, dim=0)
    batch_img = rearrange(batch_img, '(x y) c h w -> c (x h) (y w)', x=3, y=2)

    save_image(batch_img, os.path.join(save_path_grid, "geo_img.png"), normalize=True)



def pyvista_inference(args):
    input_name_list = os.listdir(args.OUTPUT_PATH)
    
    for input_name in tqdm(input_name_list, desc="PyVista inference", leave=True):
        ply_path = os.path.join(args.OUTPUT_PATH, input_name, "bini_result", "mesh_k_2.ply")
        out_path_single = os.path.join(args.OUTPUT_PATH, input_name, "pyvista_output")
        out_path_grid = os.path.join(args.OUTPUT_PATH, input_name)
        render_ply(ply_path, out_path_single, out_path_grid)
    
        