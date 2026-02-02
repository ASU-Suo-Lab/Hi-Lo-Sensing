import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from .dynamic_pillar_vfe import PFNLayerV2


class DynamicVoxelVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.pc_type = self.model_cfg.TYPE
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        if self.pc_type == 'Radar':
            points = batch_dict['radar_points'] # (batch_idx, x, y, z, i, e)
        else:
            points = batch_dict['points'] # (batch_idx, x, y, z, i, e)

        points_coords = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / self.voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1,2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        # f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        if self.pc_type == 'Radar':
             batch_dict['radar_pillar_features'] = batch_dict['radar_voxel_features'] = features
             batch_dict['radar_voxel_coords'] = voxel_coords
        else:
            batch_dict['pillar_features'] = batch_dict['voxel_features'] = features
            batch_dict['voxel_coords'] = voxel_coords

        return batch_dict


class DynamicVoxelVFE_LRFusion(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.use_norm = self.model_cfg.USE_NORM
        self.radar_proj_dim = model_cfg.RADAR_PROJ_DIM
        self.radar_proj_layer = nn.Linear(self.radar_proj_dim, self.radar_proj_dim, bias=False)
        nn.init.normal_(self.radar_proj_layer.weight, mean=0, std=0.01)
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_in = 8 + 6 + (1 if self.with_distance else 0)  # 8 base + cluster(3)+center(3) [+distance]
        self.pfn_layers = self._build_pfn_stack(pfn_in)     # 一套足够，删掉 lidar/radar 两套


        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
    
    def _build_pfn_stack(self, in_dim):
        pfn_layers = []
        num_filters = [in_dim] + list(self.num_filters)
        for i in range(len(num_filters) - 1):
            pfn_layers.append(PFNLayerV2(num_filters[i], num_filters[i+1], self.use_norm, last_layer=(i >= len(num_filters)-2)))
        return nn.ModuleList(pfn_layers)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def _build_one_modality(self, points, modality, pfn_layers=None):
        device = points.device
        points_coords = torch.floor(
             (points[:, [1, 2, 3]] - self.point_cloud_range[[0, 1, 2]]) / self.voxel_size[[0, 1, 2]]
        ).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1, 2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xyz + \
                   points_coords[:, 0] * self.scale_yz + \
                   points_coords[:, 1] * self.scale_z + \
                   points_coords[:, 2]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        # === 构造“每点 8 维语义对齐 base” ===
        if modality == 'LiDAR':
            # [x,y,z, intensity, ts, 0, 0, 0]
            base_raw = points[:, 1:6]  # x,y,z,i,ts  -> (N,5)
            zeros3 = torch.zeros((base_raw.size(0), 3), device=device, dtype=base_raw.dtype)
            base_feat = torch.cat([base_raw, zeros3], dim=1)  # (N,8)
        else:
            # Radar: [x,y,z, 0,0, Proj(vx,vy,ts)]
            radar_raw = points[:, 1:7]           # x,y,z,vx,vy,ts -> (N,6)
            proj_in   = radar_raw[:, 3:]         # vx,vy,ts -> (N,3)
            proj_out  = self.radar_proj_layer(proj_in)  # (N,3)
            zeros2 = torch.zeros((radar_raw.size(0), 2), device=device, dtype=radar_raw.dtype)
            base_feat = torch.cat([radar_raw[:, :3], zeros2, proj_out], dim=1)  # (N,8)

        # 拼 PFN 输入
        feats = [base_feat, f_cluster, f_center]
        if self.with_distance:
            pts_dist = torch.norm(points[:, 1:4], p=2, dim=1, keepdim=True)
            feats.append(pts_dist)
        pfn_in_feat = torch.cat(feats, dim=-1)  # (N, 14/15)
        # 保险断言（训练无代价）
        expected_in = 8 + 3 + 3 + (1 if self.with_distance else 0)
        assert pfn_in_feat.shape[1] == expected_in, f"PFN input dim {pfn_in_feat.shape[1]} vs expected {expected_in}"

        # PFN 前向（按点 -> voxel 聚合由 PFNLayerV2 内部 scatter 完成）
        x = pfn_in_feat
        for pfn in self.pfn_layers:
            x = pfn(x, unq_inv)  # -> (num_voxels, C_out)

        # 体素坐标
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((
            unq_coords // self.scale_xyz,
            (unq_coords % self.scale_xyz) // self.scale_yz,
            (unq_coords % self.scale_yz) // self.scale_z,
            unq_coords % self.scale_z
        ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]  # [b, z, y, x]

        return x, voxel_coords

    def forward(self, batch_dict, **kwargs):
        lidar_features, lidar_voxel_coords = self._build_one_modality(batch_dict['points'], 'LiDAR')
        radar_features, radar_voxel_coords = self._build_one_modality(batch_dict['radar_points'], 'Radar')

        voxel_features = torch.cat([lidar_features, radar_features], dim=0).contiguous()
        voxel_coords   = torch.cat([lidar_voxel_coords, radar_voxel_coords], dim=0)

        batch_dict['voxel_features'] = voxel_features
        batch_dict['voxel_coords']   = voxel_coords
        return batch_dict