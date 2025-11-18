import torch
from models.lightning_model import MVD_geo_ref_attention_LightningModel
from PIL import Image
from torchvision.transforms import v2
from torchvision import transforms as T
import numpy as np
from torchvision.transforms import InterpolationMode
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from transformers import CLIPImageProcessor
from models.geo_pipeline import MVD_geo_ref_attention_Pipeline
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from diffusers import UNet2DConditionModel

ckpt_path = r"E:\CheckPoint\zero123plus-v1.2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
   
    pipeline = MVD_geo_ref_attention_Pipeline.from_pretrained(ckpt_path).to(device)
    
    p1 = "A clean studio photo of a single subject with high detail and sharp focus, on a plain light-gray background." 
    p2 = "The subject is captured from 6 consistent multi-view angles, with soft diffuse studio lighting and neutral color grading. "
    p3 = "The object is solid, contiguous, with continuous surfaces, an unbroken silhouette, and uniform scale and proportions."
    prompt = [p1 + p2 + p3] *7
    
    input_image_clip = torch.randn(7, 3, 224, 224).to(device)
    image_embeds = pipeline.vision_encoder(input_image_clip, output_hidden_states=False).image_embeds 
    image_embeds = image_embeds.unsqueeze(-2)
    print("image_embeds.shape:", image_embeds.shape)
    
    ramp = image_embeds.new_tensor(pipeline.config.ramping_coefficients).unsqueeze(-1)
    print("ramp.shape:", ramp.shape)
    
    
    emb = pipeline.encode_text_prompt(prompt)
    print("emb.shape:", emb.shape)
    
    image_embed_ramp = ramp * image_embeds + emb
    print("image_embed_ramp.shape:", image_embed_ramp.shape)
    
    