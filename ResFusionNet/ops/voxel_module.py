# This file is modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/voxelize.py

import time
import torch
import torch.nn as nn
from .voxel_op import hard_voxelize

def adjust_half_point_five(x, tolerance=1e-9):
    """
    Adjusts the given number if its fractional part is approximately 0.5 by adding 0.001.
    
    Parameters:
    x (float): The number to be adjusted.
    tolerance (float): The tolerance for comparing the fractional part.
    
    Returns:
    float: The adjusted number.
    """
    integer_part = int(x)
    fractional_part = x - integer_part
    
    # Check if the fractional part is approximately 0.5 within a tolerance
    if abs(fractional_part - 0.5) < tolerance:
        x += 0.001
    
    return x



def dynamic_voxelize(points, temp_coors, voxel_size, coors_range, grid_size, num_points, num_features, NDim):
    # 确定每个点在哪个体素（voxel）内部，并将其坐标转换为体素坐标系中的索引
    for i in range(num_points):
        c_x = torch.floor((points[i, 0] - coors_range[0]) / voxel_size[0]).int().item()
        if c_x < 0 or c_x >= grid_size[0]:
            temp_coors[i, 0] = -1
            continue

        c_y = torch.floor((points[i, 1] - coors_range[1]) / voxel_size[1]).int().item()
        if c_y < 0 or c_y >= grid_size[1]:
            temp_coors[i, 0] = -1
            continue

        c_z = torch.floor((points[i, 2] - coors_range[2]) / voxel_size[2]).int().item()
        if c_z < 0 or c_z >= grid_size[2]:
            temp_coors[i, 0] = -1
            continue

        temp_coors[i, 0] = c_z
        temp_coors[i, 1] = c_y
        temp_coors[i, 2] = c_x


def carla_hard_voxelize(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, NDim=3, deterministic=True):
    # 初始化一些变量
    voxel_num = 0
    num_points = points.size(0)
    num_features = points.size(1)

    # 计算网格尺寸
    grid_size = torch.round(torch.tensor([adjust_half_point_five(num) for num in ((torch.tensor(coors_range[NDim:]) - torch.tensor(coors_range[:NDim])) / torch.tensor(voxel_size)).tolist()])).int().tolist()
    # grid_size = torch.round((torch.tensor(coors_range[NDim:]) - torch.tensor(coors_range[:NDim])) / torch.tensor(voxel_size)).int().tolist()

    # 创建体素坐标索引张量，映射 体素坐标索引 到 体素索引
    coor_to_voxelidx = -torch.ones((grid_size[2], grid_size[1], grid_size[0]), dtype=torch.int, device=points.device)

    # 临时坐标张量，映射 点云中的点 到 体素坐标索引
    temp_coors = torch.zeros((num_points, NDim), dtype=torch.int, device=points.device)

    # 确定每个点在哪个体素（voxel）内部，并将其坐标转换为体素坐标系中的索引，赋值到temp_coors
    dynamic_voxelize(points, temp_coors, voxel_size, coors_range, grid_size, num_points, num_features, NDim)

    for i in range(num_points):
        if temp_coors[i, 0] == -1:
            continue

        voxelidx = coor_to_voxelidx[temp_coors[i, 0], temp_coors[i, 1], temp_coors[i, 2]] # 获得该点对应的体素索引

        if voxelidx == -1: # 如果该点对应的体素索引为-1，说明该点是该体素的第一个点，这时候需要创建新的体素，要更新的是体素索引；体素坐标索引 到 体素索引 的映射；coors；num_points_per_voxel
            voxelidx = voxel_num # 相当于创建新 voxel
            if max_voxels != -1 and voxel_num >= max_voxels: # 如果体素数量超过最大体素数量，结束
                continue
            voxel_num += 1 # 体素数量加1

            coor_to_voxelidx[temp_coors[i, 0], temp_coors[i, 1], temp_coors[i, 2]] = voxelidx # 更新 体素坐标索引 到 体素索引 的映射
            coors[voxelidx] = temp_coors[i] # 把 point i 对应的体素坐标索引赋值给 coors，所以 coors 其实并不是体素坐标，而是体素坐标索引

        num = num_points_per_voxel[voxelidx] # 当前，该体素内的点的数量
        if max_points == -1 or num < max_points: # 如果体素内的点的数量小于最大点数，意味着还可以继续添加点到该体素内
            voxels[voxelidx, num] = points[i] # 把 point i 添加到该体素内，其实就是存给 voxel
            num_points_per_voxel[voxelidx] += 1 # 体素内的点的数量加1

    return voxel_num # 所以这里返回的只有和当前接收到的这些 points 相关的体素内的点，及体素的一些特征


class _Voxelization(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                points,
                voxel_size,
                coors_range,
                max_points=35,
                max_voxels=20000,
                deterministic=True):
        """convert kitti points(N, >=3) to voxels.
        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        
        # points = points[1702:1704, :]
        # print(f'points: {points.shape}\n{points}')
        
        # voxels_carla = points.new_zeros(
        #     size=(max_voxels, max_points, points.size(1)))
        # coors_carla = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        # num_points_per_voxel_carla = points.new_zeros(
        #     size=(max_voxels, ), dtype=torch.int)
        
        # carla_time_start = time.time()
        # voxel_num = carla_hard_voxelize(
        #     points.contiguous(), voxels_carla, coors_carla, num_points_per_voxel_carla, voxel_size,
        #     coors_range, max_points, max_voxels, 3, deterministic
        # )
        # select the valid voxels
        # voxels_carla = voxels_carla[:voxel_num]
        # coors_carla = coors_carla[:voxel_num].flip(-1) # (z, y, x) -> (x, y, z)
        # num_points_per_voxel_carla = num_points_per_voxel_carla[:voxel_num]
        # print(f'carla voxel_num: {voxel_num}')
        # print(f'carla_time: {time.time() - carla_time_start:.8f}')
        # print(f'voxels_carla: {voxels_carla.shape}\n{voxels_carla}')
        # print(f'coors_carla: {coors_carla.shape}\n{coors_carla}')
        # print(f'num_points_per_voxel_carla: {num_points_per_voxel_carla.shape}\n{num_points_per_voxel_carla}')
        # print(f'voxels_carla: {voxels_carla.shape}')
        # print(f'coors_carla: {coors_carla.shape}')
        # print(f'num_points_per_voxel_carla: {num_points_per_voxel_carla.shape}')
        
        voxels = points.new_zeros(
            size=(max_voxels, max_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(
            size=(max_voxels, ), dtype=torch.int)
        
        cuda_time_start = time.time()
        voxel_num = hard_voxelize(points.contiguous(), voxels, coors,
                                    num_points_per_voxel, voxel_size,
                                    coors_range, max_points, max_voxels, 3,
                                    deterministic)
        
        # print(f'cuda_time: {time.time() - cuda_time_start:.8f}')
        # print(f'voxels: {voxels.shape}\n{voxels}')
        # print(f'coors: {coors.shape}\n{coors}')
        # print(f'num_points_per_voxel: {num_points_per_voxel.shape}\n{num_points_per_voxel}')
        # print(f'voxels: {voxels.shape}')
        # print(f'coors: {coors.shape}')
        # print(f'num_points_per_voxel: {num_points_per_voxel.shape}')
        # print(f'cuda voxel_num: {voxel_num}')
        
        # select the valid voxels
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num].flip(-1) # (z, y, x) -> (x, y, z)
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        
        # print(f'voxels_out: {voxels_out.shape}\n{voxels_out}')
        # print(f'coors_out: {coors_out.shape}\n{coors_out}')
        # print(f'num_points_per_voxel_out: {num_points_per_voxel_out.shape}\n{num_points_per_voxel_out}')
        # print(f'voxels_out: {voxels_out.shape}')
        # print(f'coors_out: {coors_out.shape}')
        # print(f'num_points_per_voxel_out: {num_points_per_voxel_out.shape}')
        
        # print(torch.equal(voxels_out, voxels_carla))
        # print(torch.equal(coors_out, coors_carla))
        # print(torch.equal(num_points_per_voxel_out, num_points_per_voxel_carla))
        
        # assert False
        
        
        
        return voxels_out, coors_out, num_points_per_voxel_out


class Voxelization(nn.Module):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels,
                 deterministic=True):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
    
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input):
        """
        input: shape=(N, c)
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return _Voxelization.apply(input, self.voxel_size, self.point_cloud_range,
                                   self.max_num_points, max_voxels,
                                   self.deterministic)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', deterministic=' + str(self.deterministic)
        tmpstr += ')'
        return tmpstr
