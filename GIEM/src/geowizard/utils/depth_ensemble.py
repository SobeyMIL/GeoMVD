# A reimplemented version in public environments by Xiao Fu and Mu Hu

import numpy as np
import torch

from scipy.optimize import minimize

def inter_distances(tensors: torch.Tensor):
    """
    To calculate the distance between each two depth maps.
    """
    distances = []
    for i, j in torch.combinations(torch.arange(tensors.shape[0])):
        arr1 = tensors[i : i + 1]
        arr2 = tensors[j : j + 1]
        distances.append(arr1 - arr2)
    dist = torch.concat(distances, dim=0)
    return dist


def ensemble_depths(input_images:torch.Tensor,
                    regularizer_strength: float =0.02,
                    max_iter: int =2,
                    tol:float =1e-3,
                    reduction: str='median',
                    max_res: int=None):
    """
    输入参数：
        input_images：形状为 (n_img, h, w) 的张量，包含 n_img 个深度图，空间维度为 h（高度）和 w（宽度）。
        regularizer_strength：正则化强度（默认 0.02），平衡深度图间距离误差和输出范围约束。
        max_iter：优化最大迭代次数（默认 2）。
        tol：优化收敛容差（默认 1e-3）。
        reduction：集成方式，可选 "mean"（均值）或 "median"（中值，默认）。
        max_res：最大分辨率（可选），若指定则可能下采样深度图。
    """
    # 初始化和设备设置
    device = input_images.device
    dtype = input_images.dtype
    np_dtype = np.float32

    # 备份输入
    original_input = input_images.clone()
    # 提取批次大小，即深度图数量。
    n_img = input_images.shape[0]
    # 记录输入张量形状 (n_img, h, w)
    ori_shape = input_images.shape 
    
    if max_res is not None:
        scale_factor = torch.min(max_res / torch.tensor(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            input_images = downscaler(torch.from_numpy(input_images)).numpy()
    
    # init guess
    # 计算每个深度图的最小值，形状为 (n_img,)。
    _min = np.min(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1) # get the min value of each possible depth
    # 计算最大值，形状为 (n_img,)。
    _max = np.max(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1) # get the max value of each possible depth

    s_init = 1.0 / (_max - _min).reshape((-1, 1, 1)) #(10,1,1) : re-scale'f scale
    '''
    初始尺度：
        s_init = 1.0 / (_max - _min)：计算初始尺度，使每个深度图的范围缩放到约 [0, 1]。
        .reshape((-1, 1, 1))：调整形状为 (n_img, 1, 1)，便于后续广播。
    '''
    t_init = (-1 * s_init.flatten() * _min.flatten()).reshape((-1, 1, 1)) #(10,1,1)
    '''
    初始平移：
        t_init = (-1 * s_init.flatten() * _min.flatten())：计算平移，使最小值映射到 0（即 s * min + t = 0）。
        .reshape((-1, 1, 1))：调整形状。
    '''
    
    x = np.concatenate([s_init, t_init]).reshape(-1).astype(np_dtype) #(20,)
    '''
    将尺度和平移拼接为形状 (2*n_img,) 的向量，作为优化的初始参数。
    例如，若 n_img=10，则 x 包含 20 个参数（10 个尺度 + 10 个平移）。
    '''
    
    input_images = input_images.to(device)

    # objective function 目标优化函数
    def closure(x):
        # 获取参数向量的长度
        l = len(x)
        # 提取前半部分为尺度参数，形状 (n_img,)
        s = x[: int(l / 2)]
        # 提取后半部分为平移参数，形状 (n_img,)。
        t = x[int(l / 2) :]
        s = torch.from_numpy(s).to(dtype=dtype).to(device)
        t = torch.from_numpy(t).to(dtype=dtype).to(device)

        # 每个深度图应用变换 s * depth + t，得到对齐的深度图，形状为 (n_img, h, w)。
        transformed_arrays = input_images * s.view((-1, 1, 1)) + t.view((-1, 1, 1))
        # 计算每对对齐深度图的差值，输出形状为 (n_pairs, h, w)，其中 n_pairs = C(n_img, 2)。
        dists = inter_distances(transformed_arrays)
        # 得到均方根误差（RMSE），表示深度图间的平均距离。
        sqrt_dist = torch.sqrt(torch.mean(dists**2))

        if "mean" == reduction: # 均值
            pred = torch.mean(transformed_arrays, dim=0)
        elif "median" == reduction: # 中值
            pred = torch.median(transformed_arrays, dim=0).values
        else:
            raise ValueError

        # 预测的最小值与 0 的平方根误差
        near_err = torch.sqrt((0 - torch.min(pred)) ** 2)
        # 预测的最大值与 1 的平方根误差
        far_err = torch.sqrt((1 - torch.max(pred)) ** 2)

        # 计算距离误差
        err = sqrt_dist + (near_err + far_err) * regularizer_strength
        err = err.detach().cpu().numpy().astype(np_dtype)
        return err

    res = minimize(
        closure, x, method="BFGS", tol=tol, options={"maxiter": max_iter, "disp": False}
    )
    '''
    优化：
        minimize：使用 SciPy 的优化函数，以 BFGS 方法（准牛顿法）最小化目标函数 closure。
        x：初始参数向量（尺度和平移）。
        method="BFGS"：适合平滑目标函数的优化算法。
        tol：收敛容差。
        options={"maxiter": max_iter, "disp": False}：设置最大迭代次数，不显示优化过程。
    结果：
        res.x：优化后的参数向量。
        s = x[:int(l/2)]：提取优化后的尺度，形状 (n_img,)。
        t = x[int(l/2):]：提取平移，形状 (n_img,)。
    
    '''
    x = res.x
    l = len(x)
    s = x[: int(l / 2)]
    t = x[int(l / 2) :]

    # Prediction
    s = torch.from_numpy(s).to(dtype=dtype).to(device)
    t = torch.from_numpy(t).to(dtype=dtype).to(device)
    transformed_arrays = original_input * s.view(-1, 1, 1) + t.view(-1, 1, 1) #[10,H,W]


    if "mean" == reduction:
        aligned_images = torch.mean(transformed_arrays, dim=0)  # 对齐深度图的均值
        std = torch.std(transformed_arrays, dim=0)
        uncertainty = std

    elif "median" == reduction:
        aligned_images = torch.median(transformed_arrays, dim=0).values  # 对齐深度图的中值
        # MAD (median absolute deviation) as uncertainty indicator
        abs_dev = torch.abs(transformed_arrays - aligned_images)
        mad = torch.median(abs_dev, dim=0).values
        uncertainty = mad

    # Scale and shift to [0, 1]
    _min = torch.min(aligned_images)
    _max = torch.max(aligned_images)
    aligned_images = (aligned_images - _min) / (_max - _min)
    uncertainty /= _max - _min

    return aligned_images, uncertainty