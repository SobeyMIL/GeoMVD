# A reimplemented version in public environments by Xiao Fu and Mu Hu

import matplotlib
import numpy as np
import torch
from PIL import Image


# 为 CLIP 缩放图像
def resize_max_res(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    Args:
        img (`Image.Image`):
            PIL 图像对象，包含原始图像数据。
        max_edge_resolution (`int`):
            整数，表示调整后图像的最大边长（以像素为单位）
    Returns:
        `Image.Image`('Image.Image'):
            调整大小后的 PIL 图像对象，最大边长不超过 max_edge_resolution，且纵横比与原始图像相同。
    """
    # 获取原始尺寸
    original_width, original_height = img.size

    # 计算缩放因子: 取长边对应的缩放比例
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )
    # 计算目标尺寸
    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    # 调整图像大小
    resized_img = img.resize((new_width, new_height))
    return resized_img


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


# 调整通道维度的位置
def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc
