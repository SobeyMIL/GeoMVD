# Adapted from Marigold ：https://github.com/prs-eth/Marigold

from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from models.unet_2d_condition import UNet2DConditionModel
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from utils.image_util import resize_max_res,chw2hwc,colorize_depth_maps
from utils.colormap import kitti_colormap
from utils.depth_ensemble import ensemble_depths
from utils.normal_ensemble import ensemble_normals
from utils.batch_size import find_batch_size
import cv2

class DepthNormalPipelineOutput(BaseOutput):
    """
    Output class for GeoWizard monocular depth & normal prediction pipeline.
    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        normal_np (`np.ndarray`):
            Predicted normal map, with depth values in the range of [0, 1].
        normal_colored (`PIL.Image.Image`):
            Colorized normal map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
        depth_np: 预测出的原始深度图，以 NumPy 数组形式存储，数值范围被归一化到 [0, 1]。这是用于后续计算的主要数据。
        depth_colored: 经过色彩映射处理的深度图，是一个 PIL 图像对象，用于可视化。
        normal_np: 预测出的原始法线图，以 NumPy 数组形式存储。
        normal_colored: 可视化用的彩色法线图。
        uncertainty: 如果使用了集成（ensembling）方法，这个字段会存储一个表示不确定性的图。
                     它来自多次预测结果之间的差异（中位数绝对偏差），可以用来评估模型在某些区域的“自信程度”。
    """
    depth_np: np.ndarray
    depth_colored: Image.Image
    normal_np: np.ndarray
    normal_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class DepthNormalEstimationPipeline(DiffusionPipeline):
    # two hyper-parameters
    latent_scale_factor = 0.18215

    def __init__(self,
                 unet:UNet2DConditionModel,
                 vae:AutoencoderKL,
                 scheduler:DDIMScheduler,
                 image_encoder:CLIPVisionModelWithProjection,
                 feature_extractor:CLIPImageProcessor,
                 ):
        super().__init__()

        # 将模型组件注册到管道中，方便统一管理
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        # 初始化一个实例变量，用于缓存 CLIP 图像嵌入，避免在批次或集成处理中重复计算。
        self.img_embed = None  

    @torch.no_grad()
    def __call__(self,
                 input_image:Image,
                 denoising_steps: int = 10,
                 ensemble_size: int = 10,
                 processing_res: int = 768,
                 match_input_res:bool =True,
                 batch_size:int = 0,
                 domain: str = "indoor",
                 color_map: str="Spectral",
                 show_progress_bar:bool = True,
                 ensemble_kwargs: Dict = None,
                 ) -> DepthNormalPipelineOutput:
        # input_image:     输入的 PIL 图像。
        # denoising_steps: 扩散模型去噪循环的步数 。步数越多，结果可能越精细，但耗时也越长。
        # ensemble_size:   集成预测的次数 。运行模型多次并取平均 / 中位数，可以提高结果的稳定性和鲁棒性。
        # processing_res:  在送入模型前，图像的长边被调整到的最大分辨率 。
        # match_input_res: 是否将最终输出的尺寸调整回原始输入图像的尺寸 。
        # domain:          指定场景类型（"indoor", "outdoor", "object"），这会控制场景分布解耦器使用哪个条件 。
        # color_map:       用于深度图可视化的颜色映射表。
        # show_progress_bar: 是否显示进度条。
        
        # inherit from thea Diffusion Pipeline
        device = self.device
        # 保存输入图像的原始尺寸，以便后续可能需要将结果恢复到这个尺寸。
        input_size = input_image.size
        
        # 基本的断言检查，确保输入参数的有效性
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        # -------------------------- 图像预处理 -----------------------------
        # 将图像的长边调整到 processing_res 指定的分辨率，同时保持其原始宽高比
        if processing_res > 0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            )
        
        # 确保图像是 RGB 格式，移除可能存在的 Alpha（透明度）通道。
        input_image = input_image.convert("RGB")
        image = np.array(input_image)

        # 改变数组的维度顺序， (H, W, C) -> (C, H, W)
        rgb = np.transpose(image,(2, 0, 1))
        # 归一化 [-1, 1]
        rgb_norm = rgb / 255.0 * 2.0 - 1.0 # [0, 255] -> [-1, 1]
        # 转为指定数据类型的 Tensor
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)

        # 处理后的图像范围检查
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
        
        # ---------------------- predicting depth -------------------------
        # 为了进行集成预测，将预处理后的图像张量复制 ensemble_size 份。 [C, H, W] -> [E, C, H, W]
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        # 将多个 Tensor 打包为数据集，便于DataLoader
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        
        # 设置批量大小（batch size）默认为1。
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = 1
        # 创建一个数据加载器
        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)
        
        # 初始化两个空列表 用于收集每次推理的结果
        depth_pred_ls = []
        normal_pred_ls = []

        # 进度条开关设置
        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader

        # 迭代推理，核心为 single_infer 方法
        for batch in iterable_bar:
            (batched_image, )= batch  # [1, c=3, h, w]

            # 预测深度和法向量
            depth_pred_raw, normal_pred_raw = self.single_infer(
                input_rgb=batched_image,
                num_inference_steps=denoising_steps,
                domain=domain,
                show_pbar=show_progress_bar,
            )
            # 保存预测结果
            depth_pred_ls.append(depth_pred_raw.detach().clone())
            normal_pred_ls.append(normal_pred_raw.detach().clone())
        # 将结果沿着批次维度拼接成一个大的张量 List -> [E, C, H, W]
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze() #(10,224,768)
        normal_preds = torch.concat(normal_pred_ls, axis=0).squeeze()

        # 在执行完一个大的计算任务后清空 PyTorch 占用的缓存，释放一些显存。
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        # 合并多次预测的结果。这通常是通过取平均值或中位数来完成的，
        # 从而得到一个更平滑、更鲁棒的最终预测，
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds, **(ensemble_kwargs or {})
            )
            normal_pred = ensemble_normals(normal_preds)
        else:
            depth_pred = depth_preds
            normal_pred = normal_preds
            pred_uncert = None

        # ----------------- Post processing -----------------
        # 归一化到[0, 1]，
        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)
        
        # Convert to numpy
        depth_pred = depth_pred.cpu().numpy().astype(np.float32)
        normal_pred = normal_pred.cpu().numpy().astype(np.float32)

        # 将深度图和法线图的尺寸调整回输入图像的原始尺寸
        if match_input_res:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(input_size)
            depth_pred = np.asarray(pred_img)
            normal_pred = cv2.resize(chw2hwc(normal_pred), input_size, interpolation = cv2.INTER_NEAREST)

        # Clip output range: current size is the original size
        depth_pred = depth_pred.clip(0, 1)
        normal_pred = normal_pred.clip(-1, 1)
    
        #----------------------------- 可视化---------------------------------
        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap=color_map
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)

        normal_colored = ((normal_pred + 1)/2 * 255).astype(np.uint8)
        normal_colored_img = Image.fromarray(normal_colored)

        self.img_embed = None
        
        return DepthNormalPipelineOutput(
            depth_np = depth_pred,
            depth_colored = depth_colored_img,
            normal_np = normal_pred,
            normal_colored = normal_colored_img,
            uncertainty=pred_uncert,
        )
    
    def __encode_img_embed(self, rgb):
        """
        Encode clip embeddings for img
         为输入图像生成 CLIP 图像嵌入。
         这对应了论文中提到的，使用 CLIP 嵌入作为全局指导。
        """
        clip_image_mean = torch.as_tensor(self.feature_extractor.image_mean)[:,None,None].to(device=self.device, dtype=self.dtype)
        clip_image_std = torch.as_tensor(self.feature_extractor.image_std)[:,None,None].to(device=self.device, dtype=self.dtype)

        img_in_proc = TF.resize(
            (rgb + 1) / 2,
            (self.feature_extractor.crop_size['height'], self.feature_extractor.crop_size['width']), 
            interpolation=InterpolationMode.BICUBIC, 
            antialias=True
        )
        # do the normalization in float32 to preserve precision
        img_in_proc = ((img_in_proc.float() - clip_image_mean) / clip_image_std).to(self.dtype)        
        img_embed = self.image_encoder(img_in_proc).image_embeds.unsqueeze(1).to(self.dtype)

        self.img_embed = img_embed

        
    @torch.no_grad()
    def single_infer(self,
                     input_rgb:torch.Tensor,
                     num_inference_steps:int,
                     domain:str,
                     show_pbar:bool,
                     ):

        device = input_rgb.device

        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device) # here the numbers of the steps is only 10.
        timesteps = self.scheduler.timesteps  # [T]
        
        # VAE encode image
        rgb_latent = self.encode_RGB(input_rgb)
        
        # 扩散起点噪声，
        geo_latent = torch.randn(rgb_latent.shape, device=device, dtype=self.dtype).repeat(2, 1, 1, 1)
        rgb_latent = rgb_latent.repeat(2,1,1,1)

        # 调用 __encode_img_embed 来生成CLIP图像嵌入。
        if self.img_embed is None:
            self.__encode_img_embed(input_rgb)
        
        batch_img_embed = self.img_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 1, 768]

        # ------------------------几何切换器 (Geometry Switcher) 的实现-------------------------------
        geo_class = torch.tensor([[0., 1.], [1, 0]], device=device, dtype=self.dtype) # [2, 2]
        '''
        创建一个 [[0., 1.], [1, 0]] 的张量。这可以看作是两个独热（one-hot）编码。
        第一个 [0., 1.] 将用于标识法线，第二个 [1, 0] 将用于标识深度。
        '''
        geo_embedding = torch.cat([torch.sin(geo_class), torch.cos(geo_class)], dim=-1) # [2, 4]

        # ---------------------场景分布解耦器 (Scene Distribution Decoupler) ----------------------------
        if domain == "indoor":
            domain_class = torch.tensor([[1., 0., 0]], device=device, dtype=self.dtype).repeat(2,1) # [2, 3]
        elif domain == "outdoor":
            domain_class = torch.tensor([[0., 1., 0]], device=device, dtype=self.dtype).repeat(2,1)
        elif domain == "object":
            domain_class = torch.tensor([[0., 0., 1]], device=device, dtype=self.dtype).repeat(2,1)
        domain_embedding = torch.cat([torch.sin(domain_class), torch.cos(domain_class)], dim=-1) # [2, 6]

        class_embedding = torch.cat((geo_embedding, domain_embedding), dim=-1) # [2, 10]

        # Denoising loop
        # 进度条
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        
        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, geo_latent], dim=1) # [2, 8, h, w]

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t.repeat(2), encoder_hidden_states=batch_img_embed, class_labels=class_embedding
            ).sample  # [2, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            geo_latent = self.scheduler.step(noise_pred, t, geo_latent).prev_sample

        geo_latent = geo_latent
        torch.cuda.empty_cache()

        depth = self.decode_depth(geo_latent[0][None])
        depth = torch.clip(depth, -1.0, 1.0)
        depth = (depth + 1.0) / 2.0
        
        normal = self.decode_normal(geo_latent[1][None])
        normal /= (torch.norm(normal, p=2, dim=1, keepdim=True)+1e-5)
        normal *= -1. # 坐标系调整。

        return depth, normal
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.
        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.
        Returns:
            `torch.Tensor`: Image latent.
        """

        # encode
        h = self.vae.encoder(rgb_in) # [1, 8, 96, 96]

        moments = self.vae.quant_conv(h) # [1, 8, 96, 96]
        mean, logvar = torch.chunk(moments, 2, dim=1) # [1, 4, 96, 96]
        # scale latent
        rgb_latent = mean * self.latent_scale_factor # [1, 4, 96, 96]
        
        return rgb_latent
    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.
        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.
        Returns:
            `torch.Tensor`: Decoded depth map.
        """

        # scale latent
        depth_latent = depth_latent / self.latent_scale_factor # [1, 4, 96, 96]
        # decode
        z = self.vae.post_quant_conv(depth_latent) # [1, 4, 96, 96]
        stacked = self.vae.decoder(z) # [1, 3, 768, 768]
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True) # [1, 1, 768, 768]
        return depth_mean

    def decode_normal(self, normal_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode normal latent into normal map.
        Args:
            normal_latent (`torch.Tensor`):
                Depth latent to be decoded.
        Returns:
            `torch.Tensor`: Decoded normal map.
        """

        # scale latent
        normal_latent = normal_latent / self.latent_scale_factor # [1, 4, 96, 96]
        # decode
        z = self.vae.post_quant_conv(normal_latent) # [1, 4, 96, 96]
        normal = self.vae.decoder(z) # [1, 3, 768, 768]
        return normal
        