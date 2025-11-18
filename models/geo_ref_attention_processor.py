from typing import Optional, List, Callable

from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.schedulers import DDPMScheduler
from diffusers.schedulers import EulerAncestralDiscreteScheduler

import torch
import math
import numpy as np
from einops import rearrange
from torch import nn
import torch.nn.functional as F


'''
参考注意力处理器: 基于Pyhtorch2.0的SDPA
1. 使用参考注意力，处理参考图像
2. 使用参考注意力，处理几何特征
'''

class DecoupledGeoReferenceSelfAttnProcessor2_0(torch.nn.Module):
    """
    Attention processor for Self-Attention and Geo-Reference Attention for PyTorch 2.0.
    """
    def __init__(
        self,
        query_dim: int,
        inner_dim: int,
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        query_dim: int,                # 查询维度
        inner_dim: int,                # 内部维度
        name: Optional[str] = None,    # 名称
        enabled: bool = True,          # 是否启用当前模块
        """
        super().__init__()
        # 限定PyTorch版本为2.0，检查是否支持 scaled_dot_product_attention
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(  
                "DecoupledGeoReferenceSelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        
        self.enabled = enabled
        self.name = name
        if enabled:
            # 参考注意力，用于处理几何图像
            self.to_q_geo_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_k_geo_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_v_geo_ref = nn.Linear(
                in_features=query_dim, out_features=inner_dim, bias=False
            )
            self.to_out_geo_ref = nn.ModuleList(
                [
                    nn.Linear(in_features=inner_dim, out_features=query_dim, bias=True),
                    nn.Dropout(0.0),
                ]
            )
    '''
    三种功能：
    1. mode = "GEO" 记录几何图像的hidden_state：记为 geo_ref_hidden_state
        - 时间步为0
        - 全过程都要使用，需要一直保存在 geo_cache_dict    
        - 任何情况下都要在第一次运行，且只运行一次
        - 记录后，原本的自注意力仍然需要执行，并行参考注意力不需要执行
    2. mode = "IMG" 记录参考图像的hidden_state：记为 img_ref_hidden_state
        - 时间步跟随输入x_t
        - 每次推理重新计算，反复读写，保存在 img_cache_dict
        - 记录后，原本的自注意力仍然需要执行，并行参考注意力不需要执行
    3. mode = "FORWARD" 正常forward
        - 通过img_ref_hidden_state 和 hidden_states，计算出self attention  (pop img_ref_dict)
        - 通过geo_ref_hidden_state 计算并行注意力                          (直接读取 geo_ref_dict)
        - 将self attention和并行注意力相加，得到最终的hidden_states
    '''
        
    def geo_attention_mask(self, cell_size: int)->torch.FloatTensor:
        azimuth = [30, 90, 150, 210, 270, 330]    # 方向角
        # elevation = [30, -20, 30, -20, 30, -20]  # 仰角
        elevation = [0, 0, 0, 0, 0, 0]  # 仰角
        def get_view_distance(azimuth: List[float], elevation: List[float])->List[float]:
            cos_azimuth = [math.fabs(math.cos(math.radians(a)) ) for a in azimuth]
            cos_elevation = [math.fabs(math.cos(math.radians(e))) for e in elevation]
            cos_distance_list = [i * j for i, j in zip(cos_azimuth, cos_elevation)]        
            return cos_distance_list  
        
        cos_distance_list = get_view_distance(azimuth, elevation)
        
        mask_list = [] # 6个mask张量
        for cd in cos_distance_list:
            mask_cell = torch.ones(cell_size, cell_size, 
                                   device=self.to_q_geo_ref.weight.device, 
                                   dtype=self.to_q_geo_ref.weight.dtype) * cd
            mask_list.append(mask_cell)
        mask = torch.stack(mask_list, dim=0)
        mask = rearrange(mask, '(x y) h w -> (x h) (y w)', x=3, y=2)
        mask = torch.abs(mask)
        return mask[None, None, :, :]
    
    def get_masked_hidden_states(self, hidden_states):
        B, L, C = hidden_states.shape
        cell_size = int(math.sqrt(L // 6))
        
        mask = self.geo_attention_mask(cell_size)
        hidden_states_4d = hidden_states.reshape(B, cell_size * 3, cell_size * 2, C)
        hidden_states_4d = hidden_states_4d.permute(0, 3, 1, 2)
        hidden_states_4d = hidden_states_4d * mask
        hidden_states = hidden_states_4d.permute(0, 2, 3, 1).reshape(B, -1, C)
        return hidden_states
    
    
    def SDPA_forward(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        '''
        执行基本的注意力计算
        '''
        # 计算残差
        residual = hidden_states
        # 空间归一化
        if attn.spatial_norm is not None: # attn.spatial_norm = None
            hidden_states = attn.spatial_norm(hidden_states, temb)
        
        # 确定输入的维度数量
        input_ndim = hidden_states.ndim
        # 将输入张量展平为三个维度，便于后续计算
        #[b, c, h, w] -> [b, c, h*w] -> [b, h*w, c]
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        
        # 确定encoder_hidden_states的维度 如果encoder_hidden_states为None，则使用hidden_states的维度
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        # 准备注意力掩码
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        # 组归一化
        if attn.group_norm is not None: # attn.group_norm = None
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        # 计算 Q
        query = attn.to_q(hidden_states)
        
        # 计算 K，V
        # 如果encoder_hidden_states为None，则使用hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果encoder_hidden_states不为None，则判断是否进行归一化
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        # 确定内部维度 
        inner_dim = key.shape[-1]
        # 确定头维度
        head_dim = inner_dim // attn.heads
        
        # 调整形状，便于SDPA计算
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   =   key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # 执行SDPA计算注意力 [batch, num_heads, seq_len_q, head_dim]
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, 
            attn_mask=attention_mask, 
            dropout_p=0.0, 
            is_causal=False
        )
        
        # 调整形状，便于后续计算
        # [batch, num_heads, seq_len_q, head_dim] -> [batch, seq_len_q, num_heads * head_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        # 将hidden_states转换为query的dtype
        hidden_states = hidden_states.to(query.dtype)
        
        # 输出线性投影 [inner_dim] -> [out_dim]
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        # 恢复输入的形状
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            ) 
        # 残差连接
        if attn.residual_connection: # attn.residual_connection = True
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
    
    
    def geo_SDPA_forward(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        batch_size, sequence_length, dim = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        
        query = self.to_q_geo_ref(hidden_states)
        key = self.to_k_geo_ref(encoder_hidden_states)
        value = self.to_v_geo_ref(encoder_hidden_states)
        
        # 确定内部维度 
        inner_dim = key.shape[-1]
        # 确定头维度
        head_dim = inner_dim // attn.heads
        
        # 调整形状，便于SDPA计算
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   =   key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        
        # 调整形状，便于后续计算
        # [batch, num_heads, seq_len_q, head_dim] -> [batch, seq_len_q, num_heads * head_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out_geo_ref[0](hidden_states)
        hidden_states = self.to_out_geo_ref[1](hidden_states)
        hidden_states = self.get_masked_hidden_states(hidden_states)
      
        return hidden_states / attn.rescale_output_factor
    
    
    def __call__(
        self,
        # AttentionProcessor2_0的基本参数
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        
        # 额外功能参数
        is_cfg_guidance: bool = True, # 是否为条件引导模式
        mode: str = "FORWARD", # 可选 "GEO", "IMG", "FORWARD"
        geo_cache_dict: dict = None, # mode = "GEO" 时使用, 只用于保存hidden_states
        img_cache_dict: dict = None, # mode = "IMG" 时使用, 只用于保存hidden_states
        geo_ref_dict: dict = None, # mode = "FORWARD" 时使用, 只用于读取hidden_states
        img_ref_dict: dict = None, # mode = "FORWARD" 时使用, 只用于读取hidden_states
        geo_attention_scale = None, # mode = ""FORWARD"" 时使用, 几何注意力权重
        
    ) -> torch.FloatTensor:
        # 确定自注意力层和交叉注意力层的输入
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        if mode == "GEO":
            if self.enabled:
                # 启用的模块才需要保存hidden_states
                geo_cache_dict[self.name] = hidden_states.detach()
            return self.SDPA_forward(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        
        elif mode == "IMG": # IMG 参考在CFG启动的情况下正负样本的长度不一样，需要分离正负样本进行处理
            if self.enabled:
                img_cache_dict[self.name] = encoder_hidden_states.detach()
            return self.SDPA_forward(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        
        elif mode == "FORWARD":
            # 如果模块未启用参考注意力，则直接返回原始的SDPA计算结果
            if not self.enabled:
                return self.SDPA_forward(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            # 1. 计算图像参考自注意力           
            img_ref_hidden_states = img_ref_dict.pop(self.name)  # 从img_ref_dict中取出hidden_states
            if is_cfg_guidance:
                hs_uncond, hs_cond = hidden_states.chunk(2)
                ehs_uncond, ehs_cond = encoder_hidden_states.chunk(2)
                h0 = self.SDPA_forward(
                    attn, 
                    hs_uncond, 
                    ehs_uncond,
                    attention_mask, temb
                )
                h1 = self.SDPA_forward(
                    attn, 
                    hs_cond, 
                    torch.cat([ehs_cond, img_ref_hidden_states], dim=1),
                    attention_mask, temb
                )
                hidden_states_img = torch.cat([h0, h1], dim=0)
            else:
                hidden_states_img = self.SDPA_forward( 
                    attn, 
                    hidden_states, 
                    torch.cat([encoder_hidden_states, img_ref_hidden_states], dim=1),
                    attention_mask, temb
                )
            # 2. 计算几何参考自注意力
            geo_ref_hidden_states = geo_ref_dict[self.name]
            if is_cfg_guidance:
                hs_uncond, hs_cond = hidden_states.chunk(2)
                ehs_uncond, ehs_cond = encoder_hidden_states.chunk(2)
                gh_0 = self.geo_SDPA_forward(
                    attn, 
                    hs_uncond, 
                    ehs_uncond, 
                    attention_mask, temb
                )
                gh_1 = self.geo_SDPA_forward(
                    attn, 
                    hs_cond, 
                    geo_ref_hidden_states,
                    attention_mask, temb
                )
                hidden_states_geo = torch.cat([gh_0, gh_1], dim=0)
            else:
                hidden_states_geo = self.geo_SDPA_forward(
                    attn, 
                    hidden_states, 
                    geo_ref_hidden_states, 
                    attention_mask, temb
                )

            # 3. 合并注意力
            hidden_states = hidden_states_img * (1 - geo_attention_scale) + hidden_states_geo * geo_attention_scale
            return hidden_states





class RefNoisedImgAndGeoUNet(torch.nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        train_sched: DDPMScheduler,
        val_sched: EulerAncestralDiscreteScheduler
    ) -> None:
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched
        
        attntion_processor_dict = {}
        for name, _ in self.unet.attn_processors.items():
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            # 2. up_blocks 设置隐藏层大小为UNet上采样块的输出通道数。
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            # 3. down_blocks 设置隐藏层大小为UNet下采样块的输出通道数。
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
        
            attntion_processor_dict[name] = DecoupledGeoReferenceSelfAttnProcessor2_0(
                query_dim=hidden_size,
                inner_dim=hidden_size,
                name=name,
                enabled=name.endswith("attn1.processor"),
            )
            
        
        self.unet.set_attn_processor(attntion_processor_dict)
        self.copy_attn_weights()
        
        self.geo_ref_dict = None
        
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)
        
    def copy_attn_weights(self):
        state_dict = self.unet.state_dict()
        for key in state_dict.keys():
            if "_geo_ref" in key:
                compatible_key = key.replace("_geo_ref", "").replace("processor.", "")
                state_dict[key] = state_dict[compatible_key].clone()
        self.unet.load_state_dict(state_dict)
        
    def forward_geo(
        self, 
        geo_latent: torch.FloatTensor, 
        encoder_hidden_states, 
        is_cfg_guidance = True, 
    ):
        if is_cfg_guidance:
            ehs_uncond, ehs_cond = encoder_hidden_states.chunk(2)
            encoder_hidden_states = ehs_cond
        
        if self.training:
            timesteps = self.train_sched.timesteps
        else:
            timesteps = self.val_sched.timesteps
        
        geo_cache_dict = {}
        with torch.inference_mode():
            self.unet(
                geo_latent,
                torch.zeros_like(timesteps[0]),
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs={
                    "mode": "GEO",
                    "geo_cache_dict": geo_cache_dict,
                },
                return_dict=False,
            )
        # 二次保险：把缓存彻底切图（某些层可能没走到 detach）
        self.geo_ref_dict = {k: v.detach() for k, v in geo_cache_dict.items()}        
                
    def forward_img(
        self,
        image_latent,
        timestep,
        encoder_hidden_states,    
        img_cache_dict,
        is_cfg_guidance = True,
    ): 
        noise = torch.randn_like(image_latent)
        noisy_cond_lat = self.train_sched.add_noise(image_latent, noise, timestep)
        noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, timestep)
        
        if is_cfg_guidance:
            ehs_uncond, ehs_cond = encoder_hidden_states.chunk(2)
            encoder_hidden_states = ehs_cond
            ncl_uncond, ncl_cond = noisy_cond_lat.chunk(2)
            noisy_cond_lat = ncl_cond
            
        with torch.no_grad():
            self.unet(
                noisy_cond_lat,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=dict(mode="IMG", img_cache_dict=img_cache_dict),
            )
        
    def forward(
        self, 
        sample, 
        timestep, 
        encoder_hidden_states,
        
        cross_attention_kwargs,         # 通过这个字典传入几何条件和参考图片
                                        # - geo_latent
                                        # - image_latent
                                        # - is_cfg_guidance
        geo_attention_scale,
        *args,
        **kwargs
    ):
        image_latent = cross_attention_kwargs['image_latent']
        geo_latent = cross_attention_kwargs['geo_latent']
        is_cfg_guidance = cross_attention_kwargs['is_cfg_guidance']
        
        # 计算几何参考
        if self.geo_ref_dict is None or self.training:
            self.forward_geo(
                geo_latent,  
                encoder_hidden_states, 
                is_cfg_guidance=is_cfg_guidance,
            )
    
        # 计算图像参考
        img_cache_dict = {}
        self.forward_img(
            image_latent, 
            timestep,
            encoder_hidden_states, 
            img_cache_dict, 
            is_cfg_guidance
        )
        
        return self.unet(
            sample, 
            timestep,
            encoder_hidden_states, 
            cross_attention_kwargs=dict(
                is_cfg_guidance=is_cfg_guidance,
                mode="FORWARD",
                img_ref_dict=img_cache_dict,
                geo_ref_dict=self.geo_ref_dict,
                geo_attention_scale = geo_attention_scale
            )
        )
        
        