from rembg import remove
from PIL import Image
import io
import numpy as np

def remove_background(input_image_path, output_image_path, mask_output_path=None, resize=False):
    # 读取输入图像
    with open(input_image_path, 'rb') as input_file:
        input_data = input_file.read()

    # 移除背景
    output_data = remove(input_data)

    # 将移除背景后的数据加载为图片
    output_image = Image.open(io.BytesIO(output_data))

    # 如果需要生成掩码
    if mask_output_path is not None:
        # 获取 alpha 通道作为掩码（透明度）
        output_image_np = np.array(output_image)
        # alpha 通道是最后一维，取 alpha 通道作为掩码
        alpha_channel = output_image_np[:, :, 3]  # 获取透明度通道
        # 保存掩码为单通道灰度图像
        mask_image = Image.fromarray(alpha_channel)
        mask_image = mask_image.convert('L')  # 转换为单通道灰度图像
        mask_image.save(mask_output_path)  # 保存掩码图像

    # 如果需要裁剪并调整大小
    if resize:
        # 获取图片的尺寸
        width, height = output_image.size

        # 计算裁剪区域
        new_size = min(width, height)  # 以最短的一边为裁剪的大小
        left = (width - new_size) // 2
        top = (height - new_size) // 2
        right = (width + new_size) // 2
        bottom = (height + new_size) // 2

        # 裁剪并调整为640x640
        output_image = output_image.crop((left, top, right, bottom))  # 中心裁剪
        output_image = output_image.resize((640, 640))  # 调整为640x640

    # 保存去除背景后的图像为PNG格式
    output_image.save(output_image_path, 'PNG')


if __name__ == "__main__":
    # 使用示例
    input_image_path = 'D:\MyProject\GeoRender\example\demo_input1.png'  # 输入图像路径
    output_image_path = 'D:\MyProject\GeoRender\example\demo_input1_output.png'  # 输出图像路径
    mask_output_path = 'D:\MyProject\GeoRender\example\demo_input1_mask.png'  # 掩码输出路径

    remove_background(input_image_path, output_image_path, mask_output_path=mask_output_path, resize=True)
