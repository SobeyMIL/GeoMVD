from pytorch_lightning import LightningModule
from models.geo_pipeline import MVD_geo_ref_attention_Pipeline
from diffusers import DDPMScheduler
import torch
from torchvision.transforms import v2
import os
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch.nn.functional as F
import numpy as np
from models.geo_pipeline import RefNoisedImgAndGeoUNet


def extract_into_tensor(a, t, x_shape):
    """
    将从一维张量 a 中提取的索引为 t 的值扩展为与目标张量 x_shape 相匹配的形状
    """
    # 获取t的batch
    b, *_ = t.shape

    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def scale_latents(latents):
    """
    缩放latent数值范围
    """
    latents = (latents - 0.22) * 0.75
    return latents


def save_lora(unet: PeftModel, out_dir: str):
    """只保存 LoRA 适配器（小文件）。"""
    os.makedirs(out_dir, exist_ok=True)
    unet.save_pretrained(out_dir)
    print(f"[LoRA] adapter saved to: {out_dir}")
    

class MVD_geo_ref_attention_LightningModel(LightningModule):
    def __init__(
        self, 
        pipeline: MVD_geo_ref_attention_Pipeline,
        experiment_name="debug_model",
        log_dir="log_all",
        learning_rate=5e-5,  # 5e-5 ~ 1e-4
        lora_path=None,
        continue_step = 0
        ):
        
        super().__init__()
        self.pipeline = pipeline
    
        self.continue_step = continue_step
        self.log_dir = log_dir
        self.learning_rate = learning_rate
        self.lora_path = lora_path
        self.experiment_name = experiment_name
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.train_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        
        
        self.ramping_coefficients = pipeline.ramping_coefficients # 衰减系数
        p1 = "A clean studio photo of a single subject with high detail and sharp focus, on a plain light-gray background." 
        p2 = "The subject is captured from 6 consistent multi-view angles, with soft diffuse studio lighting and neutral color grading. "
        p3 = "The object is solid, contiguous, with continuous surfaces, an unbroken silhouette, and uniform scale and proportions."
        self.prompt = [p1 + p2 + p3]
        
        self.num_timesteps = 1000
        scale = np.geomspace(1e-5, 0.32, self.num_timesteps + 1).astype(np.float32)
        scale[: self.num_timesteps // 2] = 0.0
        self.register_buffer("geo_attention_scale_list", torch.from_numpy(scale))
        
        # self.register_schedule()
        self.prepare_unet_lora()
        self.validation_step_outputs = []
    
    def setup(self, stage=None):
        # 每个 rank 上都会被调用，确保与当前 rank 的 device 对齐
        self.pipeline.to(self.device)
        
    
    def prepare_unet_lora(self):
        adapter_name = "MV_lora"        
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
       
        device = next(self.unet.parameters()).device
        dtype  = next(self.unet.parameters()).dtype
        
        if self.lora_path is not None:
            self.unet = PeftModel.from_pretrained(
                    self.unet, 
                    self.lora_path, 
                    adapter_name=adapter_name, 
                    is_trainable=True
                )
            self.unet.set_adapter(adapter_name)
            self.unet.to(device=device, dtype=dtype)
            print("PreTrain LoRA load successfully.")
            self.unet.print_trainable_parameters()
            return
    
        # 注入 LoRA 权重
        target_modules = [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn1.processor.to_q_geo_ref",
            "attn1.processor.to_k_geo_ref",
            "attn1.processor.to_v_geo_ref",
            "attn1.processor.to_out_ref.0",
        ]
        lcfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            init_lora_weights=True,   # 近零输出初始化，避免初期 loss spike
            use_rslora=True, 
            task_type=TaskType.FEATURE_EXTRACTION,  # 对通用 nn.Module 可用
        )
        self.unet = get_peft_model(
            model = self.unet, 
            peft_config = lcfg,
            adapter_name=adapter_name,
        )
        
        self.unet.set_adapter(adapter_name)
        self.unet.to(device=device, dtype=dtype)
        print("Inject LoRA successfully.")
        self.unet.print_trainable_parameters()

        
    def save_lora(self, out_dir: str):
        """只保存 LoRA 适配器（小文件）。"""
        os.makedirs(out_dir, exist_ok=True)
        self.unet.save_pretrained(out_dir)
        print(f"[LoRA] adapter saved to: {out_dir}")

    
    # # 用于速度计算的schedule
    # def register_schedule(self):
        
    #     # replace scaled_linear schedule with linear schedule as Zero123++
    #     beta_start = 0.00085
    #     beta_end = 0.0120
    #     betas = torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)
        
    #     alphas = 1. - betas
    #     alphas_cumprod = torch.cumprod(alphas, dim=0)
    #     alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)

    #     self.register_buffer('betas', betas.float())
    #     self.register_buffer('alphas_cumprod', alphas_cumprod.float())
    #     self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.float())

    #     # calculations for diffusion q(x_t | x_{t-1}) and others
    #     self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).float())
    #     self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod).float())
    #     self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).float())
    #     self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).float())
        
        
    # # 计算 GT 速度
    # def get_v(self, x, noise, t):
    #     return (
    #         extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
    #         extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
    #     )
    
    
    @torch.no_grad()
    def get_input_image_latent(self, input_image_vae):
        dtype = next(self.pipeline.vae.parameters()).dtype
        input_image_vae = input_image_vae.to(device=self.device, dtype=dtype)
        input_image_latent = self.pipeline.vae.encode(input_image_vae).latent_dist.sample()
        return input_image_latent
    
    @torch.no_grad()
    def get_geo_or_GT_image_latent(self, geo_or_GT_image_vae):
        dtype = next(self.pipeline.vae.parameters()).dtype
        geo_or_GT_image_vae = geo_or_GT_image_vae.to(device=self.device, dtype=dtype)
        geo_or_GT_image_latent = self.pipeline.vae.encode(geo_or_GT_image_vae).latent_dist.sample() * self.pipeline.vae.config.scaling_factor
        return scale_latents(geo_or_GT_image_latent)
    
    @torch.no_grad()
    def get_text_and_image_embeds(self, text, input_image_clip):
        B = input_image_clip.shape[0]
        image_embeddings = self.pipeline.vision_encoder(input_image_clip, output_hidden_states=False).image_embeds # torch.Size([1, 1024])
        image_embeddings = image_embeddings.unsqueeze(-2) # torch.Size([B, 1, 1024])
        
        text_embeddings = self.pipeline.encode_text_prompt(text * B) # torch.Size([B, 77, 1024])
        
        ramp = image_embeddings.new_tensor(self.pipeline.config.ramping_coefficients).unsqueeze(-1).unsqueeze(0) # torch.Size([77, 1])
        encoder_hidden_states = ramp * image_embeddings + text_embeddings
        return encoder_hidden_states
    
    
    def forward(
        self, 
        noisy_latent,
        timesteps,
        text_and_image_embeds,
        
        input_image_latent,
        geo_image_latent,
    ):        
        cak = dict(
            image_latent=input_image_latent,
            geo_latent = geo_image_latent,
            is_cfg_guidance = False,
        )
        geo_scale = self.geo_attention_scale_list[timesteps].to(dtype=next(self.unet.parameters()).dtype, device=self.device)[:, None, None]
        model_pred = self.unet(
            sample=noisy_latent,
            timestep=timesteps,
            encoder_hidden_states=text_and_image_embeds,
            cross_attention_kwargs=cak,
            return_dict=False,
            geo_attention_scale=geo_scale
        )[0]
        return model_pred
    
    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)
        # 只返回 loss；日志放在 training_step 里并用 .detach()
        return loss
    
        
    def training_step(self, batch, batch_idx):
        input_image_clip = batch["input_image_clip"].to(self.device)
        input_image_vae = batch["input_image_vae"].to(self.device)
        geo_image_vae = batch["geo_image_vae"].to(self.device)
        
        # print("=-"*40)
        # print(batch["file_name"])
        # print("=-"*40)
        
        GT_image_vae = batch["GT_image_vae"].to(self.device)
        
        # 1.转为 latent 和 embeds
        input_image_latent = self.get_input_image_latent(input_image_vae).detach()
        geo_image_latent = self.get_geo_or_GT_image_latent(geo_image_vae).detach()
        GT_image_latent = self.get_geo_or_GT_image_latent(GT_image_vae).detach()
        text_and_image_embeds = self.get_text_and_image_embeds(self.prompt, input_image_clip).detach()
    
        # 2.生成噪声
        noise = torch.randn_like(GT_image_latent)
        
        # 3.随机采样一个时间步
        bsz = input_image_clip.shape[0]
        timesteps = torch.randint(0, self.num_timesteps, size=(bsz,)).long().to(self.device)

        # 4.根据时间步加噪
        noisy_latent = self.train_scheduler.add_noise(GT_image_latent, noise, timesteps)
        
        # 5.计算目标速度, (默认 v_prediction)
        target = self.train_scheduler.get_velocity(GT_image_latent, noise, timesteps)
        
        # 6.预测噪声
        model_pred = self(
            noisy_latent,
            timesteps,
            text_and_image_embeds,
            
            input_image_latent,
            geo_image_latent,
        )
        
        # 7.计算损失
        loss = self.compute_loss(model_pred, target)
        
        self.log("train/loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'])
        
        if self.global_step % 25 == 0 or self.global_step == 1:
            save_dir = os.path.join(self.log_dir, self.experiment_name, "lora_path", f'global_step_{self.global_step + self.continue_step}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_lora(self.unet, save_dir)
        
        return loss
    
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        geo_image_th = batch["geo_image_th"]
        input_image_th = batch["input_image_th"]
        GT_image_th = batch["GT_image_th"]
        
        geo_pil = v2.functional.to_pil_image(geo_image_th[0]) 
        input_pil = v2.functional.to_pil_image(input_image_th[0])
        GT_pil = v2.functional.to_pil_image(GT_image_th[0])
        
        result = self.pipeline(
            image = input_pil, 
            geo_image = geo_pil,
            prompt=self.prompt,
            guidance_scale=3.0,
            num_inference_steps=28,
            generator=torch.Generator(device=self.device).manual_seed(3407) 
        ).images[0]
        
        self.validation_step_outputs.append(result)
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        save_dir = os.path.join(self.log_dir, self.experiment_name, "val_image", f'global_step_{self.global_step}')
        os.makedirs(save_dir, exist_ok=True)
        for i, result in enumerate(self.validation_step_outputs):
            result.save(os.path.join(save_dir, f'val_sample_{i}.png'))
        self.validation_step_outputs.clear()
    

    def configure_optimizers(self):
        params = [p for p in self.unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, 
            lr=self.learning_rate,
            weight_decay=0.0,       # LoRA 通常不做 WD
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=1000, 
            eta_min=self.learning_rate/4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",    # 每次 optimizer.step() 后调度
                "frequency": 1,
                "name": "cosine_restarts",
            },
        }
        

if __name__ == "__main__":
    pass