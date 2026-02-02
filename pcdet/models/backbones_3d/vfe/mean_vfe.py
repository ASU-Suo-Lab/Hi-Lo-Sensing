import torch
from torch import nn
from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.pc_type = model_cfg.TYPE
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        if self.pc_type == 'Radar':
            voxel_features, voxel_num_points = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points']
        else:
            voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer

        if self.pc_type == 'Radar':
            batch_dict['radar_voxel_features'] = points_mean.contiguous()
        else:
            batch_dict['voxel_features'] = points_mean.contiguous()
        return batch_dict


class MeanVFE_LRFusion(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = model_cfg.IN_CHANNLE
        self.radar_proj_dim = model_cfg.RADAR_PROJ_DIM
        self.radar_proj_layer = nn.Linear(self.radar_proj_dim, self.radar_proj_dim, bias=False)
        nn.init.normal_(self.radar_proj_layer.weight, mean=0, std=0.01)

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        lidar_voxel_features, lidar_num_points,lidar_voxel_coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        lidar_points_mean = lidar_voxel_features[:, :, :].sum(dim=1, keepdim=False)
        lidar_normalizer = torch.clamp_min(lidar_num_points.view(-1, 1), min=1.0).type_as(lidar_voxel_features)
        lidar_mean = lidar_points_mean / lidar_normalizer

        radar_voxel_features, radar_voxel_num_points,radar_voxel_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
        radar_points_mean = radar_voxel_features[:, :, :].sum(dim=1, keepdim=False)
        radar_normalizer = torch.clamp_min(radar_voxel_num_points.view(-1, 1), min=1.0).type_as(radar_voxel_features)
        radar_mean = radar_points_mean / radar_normalizer
 
        N_lidar, _ = lidar_mean.shape
        N_radar, _ = radar_mean.shape
        device = lidar_voxel_features.device
        # ===== LiDAR padding: [x,y,z,intensity,ts] -> [x,y,z,intensity,ts,0,0,0] =====
        lidar_fused = torch.cat([lidar_mean, torch.zeros((N_lidar, self.radar_proj_dim), device=device)], dim=1)  # (N_lidar, 8)

        # ===== Radar padding: [x,y,z,vx,vy,ts] -> [x,y,z,0,0,vx,vy,ts] =====
        radar_proj = self.radar_proj_layer(radar_mean[:,3:])
        radar_fused = torch.cat([
            radar_mean[:, :3],                # x,y,z
            torch.zeros((N_radar, 2), device=device),  # intensity, ts 占位
            radar_proj              # vx,vy,ts
        ], dim=1)  # (N_radar, 8)

        # ===== 融合 LiDAR + Radar voxel =====
        fused_voxel_features = torch.cat([lidar_fused, radar_fused], dim=0)  # (N_lidar+N_radar, 8)
        fused_coords = torch.cat([lidar_voxel_coords, radar_voxel_coords], dim=0)  # (N_lidar+N_radar, 4)

        batch_dict['voxel_features'] = fused_voxel_features.contiguous()
        batch_dict['voxel_coords'] = fused_coords
        return batch_dict
