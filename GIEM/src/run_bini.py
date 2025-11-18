from scipy.sparse import spdiags, csr_matrix, vstack
from scipy.sparse.linalg import cg
import numpy as np
from tqdm.auto import tqdm
import time
import pyvista as pv
from PIL import Image
import cv2
import argparse, os
import warnings
warnings.filterwarnings('ignore')
from .utils import *

# 辅助函数：移动掩码
def move_left(mask): return np.pad(mask,((0,0),(0,1)),'constant',constant_values=0)[:,1:]             # 左移掩码
def move_right(mask): return np.pad(mask,((0,0),(1,0)),'constant',constant_values=0)[:,:-1]           # 右移掩码
def move_top(mask): return np.pad(mask,((0,1),(0,0)),'constant',constant_values=0)[1:,:]              # 上移掩码
def move_bottom(mask): return np.pad(mask,((1,0),(0,0)),'constant',constant_values=0)[:-1,:]          # 下移掩码
def move_top_left(mask): return np.pad(mask,((0,1),(0,1)),'constant',constant_values=0)[1:,1:]        # 上左移掩码
def move_top_right(mask): return np.pad(mask,((0,1),(1,0)),'constant',constant_values=0)[1:,:-1]      # 上右移掩码
def move_bottom_left(mask): return np.pad(mask,((1,0),(0,1)),'constant',constant_values=0)[:-1,1:]    # 下左移掩码
def move_bottom_right(mask): return np.pad(mask,((1,0),(1,0)),'constant',constant_values=0)[:-1,:-1]  # 下右移掩码


# 辅助函数：sigmoid 函数
def sigmoid(x, k=1):
    return 1 / (1 + np.exp(-k * x))

# 生成法线图中的偏导数矩阵
def generate_dx_dy(mask, nz_horizontal, nz_vertical, step_size=1):
    '''
    这个函数计算法线图中的偏导数，
    生成与每个像素相邻的四个方向（上、下、左、右）的法线分量
     ^ 垂直正方向
     |
     |
     |
     o --------> 水平正方向
    - mask: 二值掩码，指定哪些像素是有效的。
    - nz_horizontal 和 nz_vertical: 这两个数组分别表示水平方向和垂直方向上的法线分量，通常是从法线图中提取出来的。
    - step_size: 像素在世界坐标系中的大小（单位：步长），默认为 1
    
    '''
    # 计算有效像素的数量
    num_pixel = np.sum(mask)

    # 生成一个整数索引数组，其形状与掩码相同。
    pixel_idx = np.zeros_like(mask, dtype=int)
    # 为每个有效像素分配一个从 0 到 num_pixel-1 的唯一索引。
    pixel_idx[mask] = np.arange(num_pixel)

    # 创建布尔掩码，以表示上下左右 4 个方向上相邻像素的存在
    has_left_mask = np.logical_and(move_right(mask), mask) # 左移掩码与掩码相与，得到左移掩码
    has_right_mask = np.logical_and(move_left(mask), mask)
    has_bottom_mask = np.logical_and(move_top(mask), mask)
    has_top_mask = np.logical_and(move_bottom(mask), mask)

    # 提取邻域像素的法线分量
    nz_left =   nz_horizontal[has_left_mask[mask]]
    nz_right =  nz_horizontal[has_right_mask[mask]]
    nz_top =    nz_vertical[has_top_mask[mask]]
    nz_bottom = nz_vertical[has_bottom_mask[mask]]

    # 创建表示每个方向偏导数的稀疏矩阵。
    # 使用提取的法线分量和像素索引构造稀疏矩阵。
    data = np.stack([-nz_left/step_size, nz_left/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_left(has_left_mask)], pixel_idx[has_left_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_left_mask[mask].astype(int) * 2)])
    D_horizontal_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_right/step_size, nz_right/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_right_mask], pixel_idx[move_right(has_right_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_right_mask[mask].astype(int) * 2)])
    D_horizontal_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_top/step_size, nz_top/step_size], -1).flatten()
    indices = np.stack((pixel_idx[has_top_mask], pixel_idx[move_top(has_top_mask)]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_top_mask[mask].astype(int) * 2)])
    D_vertical_pos = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    data = np.stack([-nz_bottom/step_size, nz_bottom/step_size], -1).flatten()
    indices = np.stack((pixel_idx[move_bottom(has_bottom_mask)], pixel_idx[has_bottom_mask]), -1).flatten()
    indptr = np.concatenate([np.array([0]), np.cumsum(has_bottom_mask[mask].astype(int) * 2)])
    D_vertical_neg = csr_matrix((data, indices, indptr), shape=(num_pixel, num_pixel))

    # Return the four sparse matrices representing the partial derivatives for each direction.
    return D_horizontal_pos, D_horizontal_neg, D_vertical_pos, D_vertical_neg


def construct_facets_from(mask):
    # Initialize an array 'idx' of the same shape as 'mask' with integers
    # representing the indices of valid pixels in the mask.
    idx = np.zeros_like(mask, dtype=int)
    idx[mask] = np.arange(np.sum(mask))

    # Generate masks for neighboring pixels to define facets
    facet_move_top_mask = move_top(mask)
    facet_move_left_mask = move_left(mask)
    facet_move_top_left_mask = move_top_left(mask)

    # Identify the top-left pixel of each facet by performing a logical AND operation
    # on the masks of neighboring pixels and the input mask.
    facet_top_left_mask = np.logical_and.reduce((facet_move_top_mask, facet_move_left_mask, facet_move_top_left_mask, mask))

    # Create masks for the other three vertices of each facet by shifting the top-left mask.
    facet_top_right_mask = move_right(facet_top_left_mask)
    facet_bottom_left_mask = move_bottom(facet_top_left_mask)
    facet_bottom_right_mask = move_bottom_right(facet_top_left_mask)

    # Return a numpy array of facets by stacking the indices of the four vertices
    # of each facet along the last dimension. Each row of the resulting array represents
    # a single facet with the format [4, idx_top_left, idx_bottom_left, idx_bottom_right, idx_top_right].
    return np.stack((4 * np.ones(np.sum(facet_top_left_mask)),
               idx[facet_top_left_mask],
               idx[facet_bottom_left_mask],
               idx[facet_bottom_right_mask],
               idx[facet_top_right_mask]), axis=-1).astype(int)


def map_depth_map_to_point_clouds(depth_map, mask, K=None, step_size=1):
    # y
    # |  z
    # | /
    # |/
    # o ---x
    H, W = mask.shape
    yy, xx = np.meshgrid(range(W), range(H))
    xx = np.flip(xx, axis=0)

    if K is None:
        vertices = np.zeros((H, W, 3))
        vertices[..., 0] = xx * step_size
        vertices[..., 1] = yy * step_size
        vertices[..., 2] = depth_map
        vertices = vertices[mask]
    else:
        u = np.zeros((H, W, 3))
        u[..., 0] = xx
        u[..., 1] = yy
        u[..., 2] = 1
        u = u[mask].T  # 3 x m
        vertices = (np.linalg.inv(K) @ u).T * depth_map[mask, np.newaxis]  # m x 3

    return vertices


def bilateral_normal_integration(normal_map,
                                 normal_mask,
                                 k=2,
                                 depth_map=None,
                                 depth_mask=None,
                                 lambda1=0,
                                 use_K=False,
                                 step_size=1,
                                 max_iter=150,
                                 tol=1e-4,
                                 cg_max_iter=5000,
                                 cg_tol=1e-3):
    
    
    """
    normal_map：法线图，每个像素的颜色表示对应 3D 表面的法线。
    normal_mask：二值掩码，指示在 normal_map 中需要集成的区域。
    k：控制表面刚度的参数。k 值越小，表面越光滑（断层越少）。
    depth_map：初始深度图（可选），用于引导积分过程。
    depth_mask：表示 depth_map 中有效深度区域的二值掩码（可选）。
    lambda1：正则化参数，控制深度图对最终结果的影响（当提供深度图时需要）。
    K：相机内参矩阵（可选），用于透视相机模型。如果没有提供，假设为正射影相机模型。
    step_size：像素在世界坐标系中的大小（仅适用于正射影模型）。
    max_iter：最大迭代次数，默认为 150。
    tol：优化过程的相对能量变化容忍度，用于判断收敛。
    cg_max_iter：共轭梯度法的最大迭代次数，默认为 5000。
    cg_tol：共轭梯度法的容忍度，默认为 1e-3。

    返回值：
        depth_map：  经过双边法线积分过程后的深度图。
        surface：    一个用 pyvista PolyData 网格表示的从深度图重建的三维表面。
        wu_map：     一个二维图像，表示每个像素的水平平滑权重（绿色表示平滑，蓝色/红色表示间断）。
        wv_map：     一个二维图像，表示每个像素的垂直平滑权重（绿色表示平滑，蓝色/红色表示间断）。
        energy_list：优化过程中每一步的能量值列表。
    """
    
    # 为了避免混淆，我们列出代码中使用的坐标系如下：
    #
    # 像素坐标系：             相机坐标系：                   法线坐标系 (the main paper's Fig. 1 (a))
    # u                          x                              y
    # |                          |  z                           |
    # |                          | /                            o -- x
    # |                          |/                            /
    # o --- v                    o --- y                      z
    # (bottom left)
    #                        o是光心;
    #                        xy-plane是与图像平面平行的平面;
    #                        +z是观察方向.
    #
    # 输入的法线图应定义在法线坐标系中
    # 相机矩阵 K 应定义在相机坐标系中。
    # K = [[fx, 0,  cx],
    #      [0,  fy, cy],
    #      [0,  0,  1]]

    
    if use_K:
        K = np.array([
            [700,    0.0, 320.0], 
            [0.0, 933.33, 320.0], 
            [0.0,    0.0,   1.0]], dtype=np.float32) 
    else:
        K = None

    # 计算有效法线数量
    num_normals = np.sum(normal_mask)
    # 根据是否提供 K，确定是使用正射影相机模型（orthographic）还是透视相机模型（perspective）。
    projection = "orthographic" if use_K is False else "perspective"
    print(f"Running bilateral normal integration with k={k} in the {projection} case. \n"
          f"The number of normal vectors is {num_normals}.") # 打印法线数量
    
    # 将法线图从法线坐标系转换到相机坐标系
    nx = normal_map[normal_mask, 1]
    ny = normal_map[normal_mask, 0]
    nz = - normal_map[normal_mask, 2]

    # 处理透视和正射影两种情况
    if use_K:  # 透视相机模型
        img_height, img_width = normal_mask.shape[:2]

        # 生成网格坐标
        yy, xx = np.meshgrid(range(img_width), range(img_height))
        # 翻转 x 坐标，因为图像坐标系的原点在左上角，而相机坐标系的原点在图像中心。
        xx = np.flip(xx, axis=0)

        # 相机内参
        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        # 计算法线图中的法线分量
        uu = xx[normal_mask] - cx
        vv = yy[normal_mask] - cy

        nz_u = uu * nx + vv * ny + fx * nz
        nz_v = uu * nx + vv * ny + fy * nz
        del xx, yy, uu, vv
    else:  # 正射影相机模型
        nz_u = nz.copy()
        nz_v = nz.copy()

    # 生成四个偏导数矩阵（左、右、上、下），分别表示水平方向和垂直方向的法线变化。
    A3, A4, A1, A2 = generate_dx_dy(normal_mask, nz_horizontal=nz_v, nz_vertical=nz_u, step_size=step_size)

    # 构造线性系统
    A = vstack((A1, A2, A3, A4))
    b = np.concatenate((-nx, -nx, -ny, -ny))

    # 初始化优化过程的变量
    W = spdiags(0.5 * np.ones(4*num_normals), 0, 4*num_normals, 4*num_normals, format="csr")
    z = np.zeros(np.sum(normal_mask))
    energy = (A @ z - b).T @ W @ (A @ z - b)

    # 初始化优化过程的变量
    tic = time.time()
    energy_list = []
    if depth_map is not None:
        m = depth_mask[normal_mask].astype(int)
        M = spdiags(m, 0, num_normals, num_normals, format="csr")
        z_prior = np.log(depth_map)[normal_mask] if use_K else depth_map[normal_mask]

    pbar = tqdm(range(max_iter))

    # 优化循环
    for i in pbar:
        # 固定权重并求解深度
        A_mat = A.T @ W @ A
        b_vec = A.T @ W @ b
        if depth_map is not None:
            depth_diff = M @ (z_prior - z)
            depth_diff[depth_diff==0] = np.nan
            offset = np.nanmean(depth_diff)
            z = z + offset
            A_mat += lambda1 * M
            b_vec += lambda1 * M @ z_prior

        D = spdiags(1/np.clip(A_mat.diagonal(), 1e-5, None), 0, num_normals, num_normals, format="csr")  # Jacob preconditioner

        # ml = smoothed_aggregation_solver(A_mat, max_levels=4)  # AMG preconditioner, not very stable but faster than Jacob preconditioner.
        # D = ml.aspreconditioner(cycle='W')
        z, _ = cg(A_mat, b_vec, x0=z, M=D, maxiter=cg_max_iter, rtol=cg_tol)
    

        # Update the weight matrices
        wu = sigmoid((A2 @ z) ** 2 - (A1 @ z) ** 2, k)
        wv = sigmoid((A4 @ z) ** 2 - (A3 @ z) ** 2, k)
        W = spdiags(np.concatenate((wu, 1-wu, wv, 1-wv)), 0, 4*num_normals, 4*num_normals, format="csr")

        # Check for convergence
        energy_old = energy
        energy = (A @ z - b).T @ W @ (A @ z - b)
        energy_list.append(energy)
        relative_energy = np.abs(energy - energy_old) / energy_old
        pbar.set_description(
            f"step {i + 1}/{max_iter} energy: {energy:.3f} relative energy: {relative_energy:.3e}")
        if relative_energy < tol:
            break
    toc = time.time()

    print(f"Total time: {toc - tic:.3f} sec")

    # Reconstruct the depth map and surface
    depth_map = np.ones_like(normal_mask, float) * np.nan
    depth_map[normal_mask] = z

    if use_K:  # perspective
        depth_map = np.exp(depth_map)
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=K)
    else:  # orthographic
        vertices = map_depth_map_to_point_clouds(depth_map, normal_mask, K=None, step_size=step_size)

    facets = construct_facets_from(normal_mask)
    if normal_map[:, :, -1].mean() < 0:
        facets = facets[:, [0, 1, 4, 3, 2]]
    
    surface = pv.PolyData(vertices, facets)
    # surface.rotate_x(180, inplace=True)
    # surface.rotate_y(0, inplace=True)
    # surface.rotate_z(90, inplace=True)

    # In the main paper, wu indicates the horizontal direction; wv indicates the vertical direction
    wu_map = np.ones_like(normal_mask) * np.nan
    wu_map[normal_mask] = wv

    wv_map = np.ones_like(normal_mask) * np.nan
    wv_map[normal_mask] = wu

    return depth_map, surface, wu_map, wv_map, energy_list


'''
python bini_cli.py --input_path [] --output_path [] --use_K

K = np.array([
        [700,    0.0, 320.0], 
        [0.0, 933.33, 320.0], 
        [0.0,    0.0,   1.0]], dtype=np.float32) 
'''



def bini_inference(args):
    input_name_list = os.listdir(args.OUTPUT_PATH)
    
    for input_name in tqdm(input_name_list, desc="BiNI inference", leave=True):
        # 加载法线图
        normal_map = np.load(os.path.join(args.OUTPUT_PATH, input_name, "GeoWirzed_output", "normal_pred.npy"))
        normal_map[:,:,0] *= -1
        
        # 加载深度图
        depth_map = np.load(os.path.join(args.OUTPUT_PATH, input_name, "GeoWirzed_output", "depth_pred.npy"))
        scale, shift = calc_scale_shift(depth_map, normal_map)
        depth_map = depth_map * scale + shift
        
        # 加载掩码
        try:
            mask = np.array(Image.open(os.path.join(args.OUTPUT_PATH, input_name, "GeoWirzed_output", "mask.png")))[...,3].astype(bool)
        except:
            mask = np.ones(normal_map.shape[:2], bool)
        
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(
            normal_map=normal_map,
            normal_mask=mask,
            depth_map=depth_map,
            depth_mask=mask,
            k=args.k,
            use_K=args.use_K,
            max_iter=args.iter,
            tol=args.tol)
        
        
        output_path = os.path.join(args.OUTPUT_PATH, input_name, "bini_result")
        os.makedirs(output_path, exist_ok=True)
        
        
        np.save(os.path.join(output_path, "energy"), np.array(energy_list))   
        
        surface.save(os.path.join(output_path, f"mesh_k_{args.k}.ply"), binary=False)
        
        wu_map = cv2.applyColorMap((255 * wu_map).astype(np.uint8), cv2.COLORMAP_JET)
        wv_map = cv2.applyColorMap((255 * wv_map).astype(np.uint8), cv2.COLORMAP_JET)
        wu_map[~mask] = 255
        wv_map[~mask] = 255
        cv2.imwrite(os.path.join(output_path, f"wu_k_{args.k}.png"), wu_map)
        cv2.imwrite(os.path.join(output_path, f"wv_k_{args.k}.png"), wv_map)
        
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--OUTPUT_PATH", type=str, default=OUTPUT_PATH)
    parser.add_argument('--use_K', action='store_true', help="是否使用相机内参")
    parser.add_argument('--k', type=float, default=2, help="双边法线积分的参数k, 越小越平滑")
    parser.add_argument('--iter', type=np.uint, default=500, help="最大迭代次数")
    parser.add_argument('--tol', type=float, default=1e-4, help="相对能量变化容忍度")
    args = parser.parse_args()
    
    bini_inference(args)
    
    
    
    
    
    