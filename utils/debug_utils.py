import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math


# ... existing code ...

# ------------------------------------- 显示张量信息 -------------------------------------
def print_tensor_info(x, name="tensor"):
    """
    打印张量或numpy数组的详细信息
    
    Args:
        x: 输入数据，可以是torch.Tensor或numpy.ndarray
        name: 数据名称，用于显示
    """
    print(f"\n{'='*40}")
    print(f"Tensor Info: {name}")
    print(f"{'='*40}")
    
    # 基本信息
    print(f"Shape:          {x.shape if hasattr(x, 'shape') else None}")
    print(f"Data Type:      {str(x.dtype) if hasattr(x, 'dtype') else None}")
    
    # 获取数值信息
    if isinstance(x, torch.Tensor):
        # 确保tensor在CPU上
        if x.is_cuda:
            x_cpu = x.detach().cpu().numpy()
        else:
            x_cpu = x.detach().numpy()
    else:
        x_cpu = x
    
    # 检查是否有数据
    if hasattr(x_cpu, 'size') and x_cpu.size > 0:
        min_val = float(x_cpu.min())
        max_val = float(x_cpu.max())
        
        # 数值范围判断
        range_type = f'[{min_val:.3f}, {max_val:.3f}]'
        
        print(f"Range:          {range_type}")
        print(f"Min:            {min_val:.6f}")
        print(f"Max:            {max_val:.6f}")
    else:
        print(f"Range:          empty")
        print(f"Min:            None (empty tensor)")
        print(f"Max:            None (empty tensor)")
    
    print(f"{'='*40}\n")



# ------------------------------------- 显示图像 -------------------------------------
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math


def show_any_image(image, title="Image", figsize=(8, 6), max_images_per_row=4):
    """
    显示任意类型的图像，自动归一化到[0,1]后显示。
    支持torch.Tensor、numpy.ndarray、PIL.Image.Image。
    """
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    import math

    def to_numpy(img):
        if isinstance(img, torch.Tensor):
            if img.is_cuda:
                img = img.detach().cpu()
            if img.requires_grad:
                img = img.detach()
            img = img.float().numpy()
        elif isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        return img

    def normalize_img(img):
        # 灰度转3通道
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        # (C,H,W)转(H,W,C)
        if img.ndim == 3 and img.shape[0] in [1,3] and img.shape[0] < img.shape[-1]:
            img = np.transpose(img, (1,2,0))
        # 只取前3通道
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:,:,:3]
        img = img.astype(np.float32)
        minv, maxv = img.min(), img.max()
        # 归一化到[0,1]
        if minv >= 0 and maxv <= 1:
            norm = img
        elif minv >= 0 and maxv <= 255:
            norm = img / 255.0
        elif minv >= -1 and maxv <= 1:
            norm = (img + 1) / 2.0
        else:
            # 任意范围线性归一化
            if maxv > minv:
                norm = (img - minv) / (maxv - minv)
            else:
                norm = np.zeros_like(img)
        norm = np.clip(norm, 0, 1)
        return norm

    # 处理batch
    imgs = []
    if isinstance(image, torch.Tensor) and image.dim() == 4:
        for i in range(image.shape[0]):
            imgs.append(image[i])
    elif isinstance(image, np.ndarray) and image.ndim == 4:
        for i in range(image.shape[0]):
            imgs.append(image[i])
    else:
        imgs = [image]

    norm_imgs = [normalize_img(to_numpy(img)) for img in imgs]

    # 显示
    n = len(norm_imgs)
    if n == 1:
        plt.figure(figsize=figsize)
        plt.imshow(norm_imgs[0])
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        n_cols = min(max_images_per_row, n)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = np.array([axes])
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        for i, img in enumerate(norm_imgs):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            ax.imshow(img)
            ax.set_title(f'Image {i+1}')
            ax.axis('off')
        for i in range(n, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            ax.axis('off')
        plt.tight_layout()
        plt.show()




# 使用示例
if __name__ == "__main__":
    # 示例1: 创建测试图像
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 示例2: 显示numpy数组
    print("显示numpy数组图像...")
    show_any_image(test_image, "Numpy Array Image")
    
    # 示例3: 显示PIL图像
    print("显示PIL图像...")
    pil_image = Image.fromarray(test_image)
    show_any_image(pil_image, "PIL Image")
    
    # 示例4: 显示torch张量 (channels first)
    print("显示torch张量图像...")
    tensor_image = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0
    show_any_image(tensor_image, "Torch Tensor Image")
    
    # 示例5: 显示带batch维度的张量 - 现在会显示为网格
    print("显示带batch维度的张量图像（网格形式）...")
    batch_tensor = torch.stack([tensor_image] * 6, dim=0)  # 创建6张图像的batch
    show_any_image(batch_tensor, "Batch Tensor Image Grid", max_images_per_row=3)
    
    # 示例6: 显示更多图像的网格
    print("显示更多图像的网格...")
    large_batch = torch.stack([tensor_image] * 12, dim=0)  # 创建12张图像的batch
    show_any_image(large_batch, "Large Batch Image Grid", max_images_per_row=4)
    
    # 示例7: 测试tensor_info函数
    print("测试tensor_info函数...")
    
    # 测试torch.Tensor
    test_tensor = torch.randn(2, 3, 224, 224)
    print_tensor_info(test_tensor, "test_tensor")
    
    # 测试numpy数组
    test_numpy = np.random.rand(5, 10)
    print_tensor_info(test_numpy, "test_numpy")
    
    # 测试GPU张量（如果可用）
    if torch.cuda.is_available():
        gpu_tensor = torch.randn(1, 3, 64, 64).cuda()
        print_tensor_info(gpu_tensor, "gpu_tensor")
    
    # 测试特殊值
    special_tensor = torch.tensor([0, 1, -1, float('nan'), float('inf')])
    print_tensor_info(special_tensor, "special_tensor")
    
    # 快速信息显示
    print("快速信息显示:")
    quick_tensor_info(test_tensor, "test_tensor")
    quick_tensor_info(test_numpy, "test_numpy")