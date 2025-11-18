from PIL import Image
import torch
import numpy as np
from einops import rearrange
from torchvision.utils import save_image
import os

image_path = r"D:\MyProject\MV_Diffusion\Top_Show_renderning\zaku_ii_UC0079\output_lora_10000_stp50.png"
out_path   = os.path.join(os.path.dirname(image_path), "line.png")

# 读图 -> tensor [C,H,W], 0~1
img = Image.open(image_path).convert("RGB")
arr = np.asarray(img)
t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # CHW

# 3 行 × 2 列
rows, cols = 3, 2
C, H, W = t.shape
tile_h, tile_w = H // rows, W // cols

# 防止不是整倍数：裁掉边缘对齐到网格
t = t[:, :tile_h * rows, :tile_w * cols]

# 用 einops 把 3×2 重排成 1×6
# CHW -> C h (rows*cols*w)
t_row = rearrange(t, 'ch (r h) (c w) -> ch h (r c w)', r=rows, c=cols)

save_image(t_row, out_path)
print(t_row.shape, 'saved to:', out_path)