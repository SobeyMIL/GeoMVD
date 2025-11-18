import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
from transformers import CLIPImageProcessor
'''
glb_name:
    - input.png: 输入单张图片     640 x 640 x 4     clip     
        -> 512 x 512 x 3
    - geo_img.png: 输入几何图片   1920 x 1280 x 3   vae
        -> 960 x 640 x 3
    - GT.png: 从glb文件渲染的图片  1920 x 1280 x 4   vae
        -> 960 x 640 x 3
'''

class GeoPipelineDataset(Dataset):
    def __init__(
        self, 
        pipeline_input_path, 
        feature_extractor_clip:CLIPImageProcessor,
        feature_extractor_vae:CLIPImageProcessor,
        geo_processor_vae:CLIPImageProcessor,
        validation=False, 
        bg_color=[1., 1., 1.]
    ):
        self.pipeline_input_path = pipeline_input_path
        self.validation = validation
        self.bg_color = bg_color
        
        self.glb_name_list = os.listdir(self.pipeline_input_path)
        self.glb_name_list = self.glb_name_list[-8:] if validation else self.glb_name_list[:-8]
        
        self.feature_extractor_clip = feature_extractor_clip
        self.feature_extractor_vae = feature_extractor_vae
        self.geo_processor_vae = geo_processor_vae
        
        self.geo_transform = A.Compose([
            A.Resize(960, 640, interpolation=Image.BICUBIC),
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ToTensorV2()
        ])
        self.input_transform = A.Compose([
            A.Resize(512, 512, interpolation=Image.BICUBIC),
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ToTensorV2()
        ])
        
        
        
    def __len__(self):
        return len(self.glb_name_list)
    
    def __getitem__(self, idx):
        glb_name_path = os.path.join(self.pipeline_input_path, self.glb_name_list[idx])
        input_path = os.path.join(glb_name_path, "input.png")
        geo_img_path = os.path.join(glb_name_path, "geo_img.png")
        GT_path = os.path.join(glb_name_path, "GT.png")
        
        input_image_np = self.convert_rgba_to_rgb_np(input_path)
        geo_image_np = self.convert_rgba_to_rgb_np(geo_img_path)
        GT_image_np = self.convert_rgba_to_rgb_np(GT_path)
        
        
        input_image_clip = self.feature_extractor_clip(images = input_image_np, return_tensors="pt").pixel_values[0]
        input_image_vae = self.feature_extractor_vae(images = input_image_np, return_tensors="pt").pixel_values[0]
        geo_image_vae = self.geo_processor_vae(images = geo_image_np, return_tensors="pt").pixel_values[0]
        GT_image_vae = self.geo_processor_vae(images = GT_image_np, return_tensors="pt").pixel_values[0]
    
        if self.validation:
            return {
                "geo_image_th": self.geo_transform(image = geo_image_np)["image"],
                "input_image_th": self.input_transform(image = input_image_np)["image"],
                "GT_image_th": self.geo_transform(image = GT_image_np)["image"],
                
            }
        
        return {
            "input_image_clip": input_image_clip,
            "input_image_vae": input_image_vae,
            "geo_image_vae": geo_image_vae,     # tensor(-0.6250) tensor(0.6250), torch.Size([4, 3, 960, 640])
            "GT_image_vae": GT_image_vae,       # tensor(-0.6250) tensor(0.6250), torch.Size([4, 3, 960, 640])
            "file_name": glb_name_path
        }
        
        
        
    def convert_rgba_to_rgb_np(self, image_path):
        if self.bg_color is None:
            color = [1., 1., 1.]
        else:
            color = self.bg_color
        
        pil_img = Image.open(image_path)
        
        if pil_img.mode == "RGB":
            return np.array(pil_img)
        
        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)
        image = (image * 255.).astype(np.uint8)    
        return image
    
    
def convert_rgba_to_rgb(image_path):
    color = [1., 1., 1.]
    pil_img = Image.open(image_path)

    if pil_img.mode == "RGB":
        return np.array(pil_img)

    image = np.asarray(pil_img, dtype=np.float32) / 255.
    alpha = image[:, :, 3:]
    image = image[:, :, :3] * alpha + color * (1 - alpha)
    image = (image * 255.).astype(np.uint8)    
    return image
        
def print_info(image):
    print(image.shape)
    print(image.dtype)
    print(image.min(), image.max())
   
from torchvision.utils import make_grid
def show_grid_images(image:torch.Tensor):
    image = make_grid(image, nrow=4)
    image_pil = v2.functional.to_pil_image(image)
    image_pil.show()
   
from torch.utils.data import DataLoader
from tqdm import tqdm
if __name__ == "__main__":
    feature_extractor_clip = CLIPImageProcessor.from_pretrained(r"/mnt/wq/AAAAAAAAAAAAAAAAAAA_LAB/ckpt/zero123plus-v1.2/feature_extractor_clip")
    feature_extractor_vae = CLIPImageProcessor.from_pretrained(r"/mnt/wq/AAAAAAAAAAAAAAAAAAA_LAB/ckpt/zero123plus-v1.2/feature_extractor_vae")
    geo_processor_vae = CLIPImageProcessor.from_pretrained(r"/mnt/wq/AAAAAAAAAAAAAAAAAAA_LAB/ckpt/zero123plus-v1.2/geo_processor_vae")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = GeoPipelineDataset(
        pipeline_input_path=r"/mnt/wq/AAAAAAAAAAAAAAAAAAA_LAB/ObjaverseRendering/Train",
        feature_extractor_clip=feature_extractor_clip,
        feature_extractor_vae=feature_extractor_vae,
        geo_processor_vae=geo_processor_vae,
        validation=False,
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, batch in enumerate(data_loader):
        print(f"=================={i}==================")
        input_image_clip = batch["input_image_clip"].to(device) # torch.Size([4, 3, 224, 224]) [-1.7923 1.8762]
        input_image_vae = batch["input_image_vae"].to(device)   # torch.Size([4, 3, 512, 512]) [-0.6250 0.6250]
        geo_image_vae = batch["geo_image_vae"].to(device)       # torch.Size([4, 3, 960, 640]) [-0.6250 0.6250]
        GT_image_vae = batch["GT_image_vae"].to(device)         # torch.Size([4, 3, 960, 640]) [-0.6250 0.6250]

        print("name:", batch["file_name"])
        print_info(geo_image_vae)
        b, c, h, w = geo_image_vae.shape
        if (h != 960) or (w != 640):
            print("Error " * 50)



    
    
    