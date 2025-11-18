import torch
from torch import nn
from torchvision import transforms
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.models import UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDPMScheduler, EulerAncestralDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from peft import PeftModel
import numpy as np
import os
from PIL import Image
from tqdm.auto import tqdm
from typing import Optional

from models.geo_ref_attention_processor import RefNoisedImgAndGeoUNet



def to_rgb_image(maybe_rgba: Image.Image) -> Image.Image:
    if maybe_rgba.mode == 'RGB': # 如果图像模式为RGB，则直接返回
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA': # 如果图像模式为RGBA，则将图像转换为RGB
        rgba = maybe_rgba
        img = np.random.randint(255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB') # 将随机生成的图像转换为RGB模式
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class MVD_geo_ref_attention_Pipeline(DiffusionPipeline):
    latent_scale_factor = 0.18215
    ramping: nn.Linear
    depth_transforms_multi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(
        self, 
        unet: UNet2DConditionModel,
        scheduler: EulerAncestralDiscreteScheduler,
        
        vae: AutoencoderKL, 
        feature_extractor_vae: CLIPImageProcessor, 
        geo_processor_vae: CLIPImageProcessor,

        text_encoder: CLIPTextModel, 
        tokenizer: CLIPTokenizer, 

        vision_encoder: CLIPVisionModelWithProjection, 
        feature_extractor_clip: CLIPImageProcessor, 

        ramping_coefficients: Optional[list] = None,
        safety_checker=None,
        ):
        super().__init__()

        # 注册模型组件
        self.register_modules(
            unet=unet,
            scheduler=scheduler,

            vae=vae,    
            feature_extractor_vae=feature_extractor_vae,
            geo_processor_vae=geo_processor_vae,

            text_encoder=text_encoder,
            tokenizer=tokenizer,

            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,

            safety_checker=safety_checker,
        )
        
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        # VAE 的缩放因子
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.train_sched = DDPMScheduler.from_config(self.scheduler.config)
        
        self.prepare()
        
    def unscale_latents(self, latents):
        latents = latents / 0.75 + 0.22
        return latents
    
    def scale_latents(self, latents):
        latents = (latents - 0.22) / 0.75
        return latents
    
    def unscale_image(self, image):
        image = image / 0.5 * 0.8
        return image
    
    def scale_image(self, image):
        image = image * 0.5 / 0.8
        return image

    def prepare(self):
        # 训练采样器：DDPM
        self.unet = RefNoisedImgAndGeoUNet(self.unet, self.train_sched, self.scheduler).eval()
    

    def inject_lora(self, lora_path=None, is_trainable=True):
        print("Injecting LoRA...")
        if lora_path is None:
            return
        self.unet = PeftModel.from_pretrained(
            self.unet,
            lora_path,
            adapter_name="MV_lora",
            is_trainable=is_trainable,
        )
        self.unet.set_adapter("MV_lora")
        print("LoRA injected successfully.")
        self.unet.print_trainable_parameters()
    

    @torch.inference_mode()
    def encode_condition_image(self, image: torch.Tensor):
        """
        将图像编码为latent
        image: [B, 3, H, W]
        return: [B, 4, H/8, W/8]
        """
        latent = self.vae.encode(image).latent_dist.sample()
        return latent
    
    @torch.inference_mode() # 处理输出的多视角图像
    def encode_geo_images(self, images):
        dtype = self.vae.dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        posterior = self.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        latents = self.scale_latents(latents)
        return latents
    
    @torch.inference_mode()
    def decode_latents(self, latents, generator=None):
        generator = torch.Generator(device=self.device).manual_seed(0) if generator is None else generator
        latents = self.unscale_latents(latents)
        image = self.vae.decode(latents / self.vae.config.scaling_factor, generator=generator).sample
        image = self.unscale_image(image)
        
        image = self.image_processor.postprocess(image, output_type="pil")[0]
        # image.show()
        return image
    
    @torch.inference_mode()
    def encode_text_prompt(self, prompt: str):
        """
        将文本提示编码为 embedding
        Returns:
            torch.Tensor: [B=1, seq_len, hidden_size]
        """
        # 确保有 pad_token（有些 CLIP tokenizer 默认无 pad）
        if self.tokenizer.pad_token is None and hasattr(self.tokenizer, "eos_token"):
            self.tokenizer.pad_token = self.tokenizer.eos_token

        enc = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # （可选）提示截断告警
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        # print(f"prompt length is {len(ids)}")
        if len(ids) > self.tokenizer.model_max_length:
            # 这里改为你项目的日志/告警方式
            print(f"[warn] prompt truncated to {self.tokenizer.model_max_length} tokens.")

        device = next(self.text_encoder.parameters()).device
        input_ids = enc.input_ids.to(device, non_blocking=True)
        attention_mask = enc.attention_mask.to(device, non_blocking=True) if "attention_mask" in enc else None

        # 直接前向；不再套 no_grad()
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = outputs[0]  # last_hidden_state: [1, seq_len, hidden_size]

        # 如果后续会做原地操作，可解开推理标记：
        text_embeddings = text_embeddings.clone()
        return text_embeddings

    
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 0.0 and self.unet.config.time_cond_proj_dim is None

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        geo_image: Image.Image,

        prompt = "",
        guidance_scale=7.0,
        num_inference_steps: Optional[int] = 28,
        generator: Optional[torch.Generator] = None,

        width: Optional[int] = 640,
        height: Optional[int] = 960,

        output_type: Optional[str] = "pil",
        num_images_per_prompt: Optional[int] = 1,
        return_dict: Optional[bool] = True,
        *args,
        **kwargs
        ):
        """
        image: PIL Image： 输入图像
        prompt: str： 文本提示
        guidance_scale: float： CFG 引导尺度
        num_inference_steps: int： 推理步数
        width: int： 生成图像宽度
        height: int： 生成图像高度
        output_type: str： 输出图片类型
        num_images_per_prompt: int： 每个文本提示生成的图像数量
        return_dict: bool： 是否返回字典
        *args, **kwargs： 其他参数
        """
        batch_size = 1
        self._guidance_scale = guidance_scale
        

        # ---------------------------------- 处理输入图像 --------------------------------
        assert image is not None, "Condition Image is required"
        assert isinstance(image, Image.Image), "Condition Image must be a PIL Image"
        # 将图像转换为RGB模式
        image = to_rgb_image(image) # pil
        # 将图像转换为torch张量
        image_vae = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values  # torch.Size([1, 3, 512, 512])
        image_clip = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values# torch.Size([1, 3, 224, 224])
        image_vae = image_vae.to(self.vae.device, dtype=self.vae.dtype)
        image_clip = image_clip.to(self.vae.device, dtype=self.vae.dtype)

        # vae 编码
        image_latent = self.encode_condition_image(image_vae) # torch.Size([1, 4, 64, 64])

        # 如果CFG引导尺度大于1，则添加负向条件 [1, 4, 64, 64] -> [2, 4, 64, 64]
        if self.do_classifier_free_guidance():
            negative_latent = self.encode_condition_image(torch.zeros_like(image_vae).to(self.vae.device, dtype=self.vae.dtype))
            image_latent = torch.cat([negative_latent, image_latent])
    
        # ---------------------------------- 计算全局 embedding --------------------------------
        # clip 编码
        image_embeddings = self.vision_encoder(image_clip, output_hidden_states=False).image_embeds # torch.Size([1, 1024])
        image_embeddings = image_embeddings.unsqueeze(-2) # torch.Size([1, 1, 1024])
        
        # 计算全局 embedding 即 encoder_hidden_states
        # ramping_coefficients 用于调整图像特征的权重，使其与文本特征融合
        ramp = image_embeddings.new_tensor(self.config.ramping_coefficients).unsqueeze(-1) # torch.Size([77, 1])
        text_embeddings = self.encode_text_prompt(prompt) # torch.Size([1, 77, 1024])
        encoder_hidden_states = text_embeddings + image_embeddings * ramp # torch.Size([1, 77, 1024])
        if self.do_classifier_free_guidance():
            negative_embeddings = self.encode_text_prompt(" ")
            encoder_hidden_states = torch.cat([negative_embeddings, encoder_hidden_states]) # torch.Size([2, 77, 1024])
        

        # ---------------------------------- 处理几何条件图像 --------------------------------
        assert geo_image is not None, "Geometry Image is required"
        assert isinstance(geo_image, Image.Image), "Geometry Image must be a PIL Image"
        w, h = geo_image.size
        assert h * 2 == w * 3, f"geo_image的高宽比应为3:2（实际为{h}:{w}）"
        # 将图像转换为RGB模式
        geo_image = to_rgb_image(geo_image) # pil
        # 将图像转换为torch张量, 缩放尺寸，缩放数值范围
        geo_image_vae = self.geo_processor_vae(images=geo_image, return_tensors="pt").pixel_values
        geo_image_vae = geo_image_vae.to(self.vae.device, dtype=self.vae.dtype)
        # vae编码几何条件图像
        geo_image_latent = self.encode_geo_images(geo_image_vae)
 

        # cross_attention_kwargs, 用于注意力参考
        cak = dict(
            image_latent = image_latent,
            geo_latent = geo_image_latent,
            is_cfg_guidance = self.do_classifier_free_guidance(),
        )


        # --------------------------去噪循环--------------------------------
        # 设置时间步
        self.scheduler.set_timesteps(num_inference_steps, device=self.device) 
        timesteps = self.scheduler.timesteps.long() # torch.Size([28])

        iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        # 创建初始噪声
        # 计算latents的形状
        latents_shape = (
            batch_size, 
            self.unet.config.in_channels, 
            height // self.vae_scale_factor, 
            width // self.vae_scale_factor
        ) # (1, 4, 120, 80)
        # 创建噪声生成器
        noise_generator = torch.Generator(device=self.device).manual_seed(0) if generator is None else generator    
        # 创建噪声
        latents = torch.randn(
            latents_shape, 
            device=self.device, 
            dtype=self.unet.dtype,
            generator=noise_generator,
        )

        noise_pred = None
            
        # ---------------------------------------几何条件加权-------------------------------------------------
        geo_attention_scale_list = np.geomspace(1e-5, 0.32, 1000 + 1)
        # geo_attention_scale_list[:1000//2] = 0.0
        geo_attention_scale_list = torch.tensor(geo_attention_scale_list, dtype=latents.dtype, device=self.device)


        for i, t in iterable:   
            # 扩展latents，如果使用CFG [neg, pos]
            latent_model_input = torch.cat([latents, latents]) if self.do_classifier_free_guidance() else latents
            # 缩放latent_model_input，用于模型输入
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            t = torch.tensor([t], dtype=torch.long, device=self.device)

            
            # 预测噪声残差
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cak,
                return_dict=False,
                geo_attention_scale=geo_attention_scale_list[t][:, None, None]
            )[0]

            
            if self.do_classifier_free_guidance():
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self._guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 计算上一步的含噪样本
            latents = self.scheduler.step(noise_pred, t, latents, ).prev_sample
        
        # --------------------------解码图像--------------------------
        latents = self.unscale_latents(latents)
        image = self.vae.decode(latents / self.vae.config.scaling_factor, generator=generator).sample
        image = self.unscale_image(image)
        
        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)

        
 