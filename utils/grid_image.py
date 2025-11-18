import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import save_image


# def grid_image(mv_path, save_grid_path):
#     mv_img_list = os.listdir(mv_path)
#     mv_img_list = [img for img in mv_img_list if img.endswith(".png")]
#     tr_list = [
#         torch.from_numpy(np.array(Image.open(os.path.join(mv_path, mv_img)).convert("RGB"))).permute(2, 0, 1)
#         for mv_img in mv_img_list
#     ]  
#     batch_img = torch.stack(tr_list, dim=0).float()
#     batch_img = rearrange(batch_img, '(x y) c h w -> c (x h) (y w)', x=3, y=2)
#     save_image(batch_img, os.path.join(save_grid_path, "GT.png"), normalize=True)

def grid_image(mv_path, save_grid_path):
    mv_img_list = os.listdir(mv_path)
    mv_img_list = [img for img in mv_img_list if img.endswith(".png")]
    
    tr_list = [
        torch.from_numpy(np.array(Image.open(os.path.join(mv_path, mv_img)).convert("RGBA"))).permute(2, 0, 1)
        for mv_img in mv_img_list
    ]  
    
    batch_img = torch.stack(tr_list, dim=0).float()  # shape: [N, 4, H, W]
    batch_img = rearrange(batch_img, '(x y) c h w -> c (x h) (y w)', x=3, y=2)
    # 保存时指定 normalize=False，因为 normalize 会把 alpha 通道压缩到 [0,1] 可能不想要
    save_image(batch_img / 255.0, os.path.join(save_grid_path, "GT.png"), normalize=False)
    
    
if __name__ == "__main__":
    pipeline_input = "E:\Dataset\GSO\Pipeline_Input"
    gt_path = "E:\Dataset\GSO\selected_renderning\zero123++"

    for glb_name in os.listdir(pipeline_input):
        mv_path = os.path.join(gt_path, glb_name)
        save_grid_path = os.path.join(pipeline_input, glb_name)
        grid_image(mv_path, save_grid_path)