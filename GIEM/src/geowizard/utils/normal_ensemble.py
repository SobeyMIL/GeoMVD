# A reimplemented version in public environments by Xiao Fu and Mu Hu

import numpy as np
import torch

def ensemble_normals(input_images:torch.Tensor):
    normal_preds = input_images

    bsz, d, h, w = normal_preds.shape
    normal_preds = normal_preds / (torch.norm(normal_preds, p=2, dim=1).unsqueeze(1) + 1e-5)
    # 归一化
    # 沿通道维度（dim=1）计算法线向量的 L2（欧几里得）范数，得到形状为 (bsz, h, w) 的张量，表示每个像素处法线向量的模。
    # 将范数张量扩展为 (bsz, 1, h, w)
    # 将每个法线向量除以其模，归一化为单位向量

    phi = torch.atan2(normal_preds[:, 1, :, :], normal_preds[:, 0, :, :]).mean(dim=0)
    '''
    Phi（方位角）：
        normal_preds[:,1,:,:] 提取法线向量的 y 分量，形状为 (bsz, h, w)。
        normal_preds[:,0,:,:] 提取 x 分量。
        torch.atan2(y, x) 计算 y/x 的反正切，返回角度 phi，范围为 [-π, π]，表示法线向量在 xy 平面上的方位角。
        .mean(dim=0) 沿批次维度取平均，得到形状为 (h, w) 的单一角度图。
    '''
    theta = torch.atan2(torch.norm(normal_preds[:,:2,:,:], p=2, dim=1), normal_preds[:,2,:,:]).mean(dim=0)
    '''
    Theta（极角）：
        normal_preds[:,:2,:,:] 选择 x 和 y 分量，形状为 (bsz, 2, h, w)。
        torch.norm(..., p=2, dim=1) 计算 xy 平面投影的 L2 范数，得到形状为 (bsz, h, w) 的张量，表示 xy 平面上的模。
        normal_preds[:,2,:,:] 提取 z 分量。
        torch.atan2(xy_norm, z) 计算极角 theta，即法线向量与 z 轴的夹角。输入是 xy 平面模和 z 分量，因此 theta = arctan(sqrt(x² + y²) / z)。
        .mean(dim=0) 沿批次维度取平均，得到形状为 (h, w) 的角度图。
    '''
    # ----------------------- 重建共识法线图 -------------------------------
    normal_pred = torch.zeros((d, h, w)).to(normal_preds)
    normal_pred[0, :, :] = torch.sin(theta) * torch.cos(phi)
    normal_pred[1, :, :] = torch.sin(theta) * torch.sin(phi)
    normal_pred[2, :, :] = torch.cos(theta)

    angle_error = torch.acos(torch.clip(torch.cosine_similarity(normal_pred[None], normal_preds, dim=1),-0.999, 0.999))
    """
    余弦相似度：
        normal_pred[None]    为 normal_pred 添加批次维度，形状变为 (1, d, h, w)，
                            以便与 normal_preds（形状 (bsz, d, h, w)）广播。
        torch.cosine_similarity(normal_pred[None], normal_preds, dim=1) 
                            沿通道维度（dim=1）计算共识法线图与每个输入法线图的余弦相似度，
                            输出形状为 (bsz, h, w)，每个元素表示对应法线向量之间的余弦值。
        torch.clip(..., -0.999, 0.999) 将余弦值限制在 [-0.999, 0.999]，避免 acos 函数的数值问题（acos 要求输入在 [-1, 1]）。
    角度误差：    
        torch.acos(...) 将裁剪后的余弦值转换为角度误差（以弧度为单位），输出形状为 (bsz, h, w)，
                        表示共识法线与每个输入法线在每个像素处的夹角。
                        
    """
    normal_idx = torch.argmin(angle_error.reshape(bsz,-1).sum(-1))
    '''
    选择最佳法线图：
        angle_error.reshape(bsz, -1) 将角度误差张量重塑为 (bsz, h*w)，展平空间维度。
        .sum(-1) 计算每个法线图的总角度误差，得到形状为 (bsz,) 的张量。
        torch.argmin(...) 找到总角度误差最小的法线图的索引。
        normal_idx 是批次中与共识法线图最接近的法线图的索引（范围 [0, bsz-1]）。
        目的：通过计算每个输入法线图与共识法线图的像素级角度偏差，找到总偏差最小的法线图作为代表。
    '''
    # 返回批次中索引为 normal_idx 的法线图，形状为 (d, h, w)，即与共识法线图角度误差最小的法线图。
    return normal_preds[normal_idx]