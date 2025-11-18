import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import argparse
from pytorch_lightning import seed_everything
from PIL import Image
import torch
import time
from models.geo_pipeline import MVD_geo_ref_attention_Pipeline


if __name__ == "__main__":
    ckpt_path = r"E:\CheckPoint\GeoMVD"
    pipeline_input = "D:\MyProject\GeoMVD\GIEM\OUTPUT1"
    
    p1 = "A clean studio photo of a single subject with high detail and sharp focus, on a plain light-gray background." 
    p2 = "The subject is captured from 6 consistent multi-view angles, with soft diffuse studio lighting and neutral color grading. "
    p3 = "The object is solid, contiguous, with continuous surfaces, an unbroken silhouette, and uniform scale and proportions."
    prompt = p1 + p2 + p3
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline_input", type=str, default=pipeline_input)
    parser.add_argument("--save_tag", type=str, default=f"lora_10000_stp30")
    parser.add_argument("--ckpt_path", type=str, default=ckpt_path)
    lora_path = os.path.join(ckpt_path, "LoRA_ckpt", "MV_lora")
    parser.add_argument("--lora_path", type=str, default=lora_path)
    parser.add_argument("--prompt", type=str, default=prompt)
    parser.add_argument("--guidance_scale", type=float, default=3.3)
    parser.add_argument("--num_inference_steps", type=int, default=70)
    parser.add_argument("--seed", type=int, default=3047)
    
    args = parser.parse_args()
    
    

    seed_everything(args.seed)
    # 加载pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = MVD_geo_ref_attention_Pipeline.from_pretrained(args.ckpt_path)
    pipeline.inject_lora(lora_path=args.lora_path, is_trainable=False)
    pipeline = pipeline.to(device=device, dtype=torch.float16)
    # 总计时开始
    total_start_time = time.time()
    glb_list = os.listdir(args.pipeline_input)
    glb_list = sorted(glb_list)

    
    for i, glb_name in enumerate(glb_list):
        loop_start_time = time.time()  # 单个循环计时开始
        
        image_path = os.path.join(args.pipeline_input, glb_name, "input.png")
        geo_image_path = os.path.join(args.pipeline_input, glb_name, "geo_img.png")
        save_path = os.path.join(args.pipeline_input, glb_name)        # 读取图像
        image = Image.open(image_path)
        geo_image = Image.open(geo_image_path)
        
        # 生成图像              
        result = pipeline(
            image, 
            geo_image,
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(args.seed),
            width=640,
            height=960,
        ).images[0]
        
        # 保存图像
        result.save(os.path.join(save_path, f"output_{args.save_tag}.png"))
        # 显示图像
        # result.show()
        
        loop_end_time = time.time()  # 单个循环计时结束
        print(f"Loop {save_path} finished in {loop_end_time - loop_start_time:.2f} seconds")
        print(f"Sample {i} save in {save_path}")
    
    # 总计时结束
    total_end_time = time.time()
    print(f"All tasks finished in {total_end_time - total_start_time:.2f} seconds")
