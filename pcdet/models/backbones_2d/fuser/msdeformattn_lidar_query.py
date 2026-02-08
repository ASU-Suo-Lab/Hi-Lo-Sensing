import torch
from torch import nn
from einops import rearrange
from pcdet.ops.ms_deform_attn import MSDeformAttn
from .msdeformattn_bi_query import LRMSDeformAttnFuser,LearnedPositionalEncoding3D



class LRMSDeformAttnFuser_LiDAR(LRMSDeformAttnFuser):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg=model_cfg)
        self.DeformAttn1 = MSDeformAttn(d_model=self.radar_post_dim + self.lidar_post_dim, d_v=self.lidar_post_dim,
                                        n_levels=1, n_heads=8,
                                        n_points=8)  # d_model=256, n_levels=1, n_heads=8, n_points=4
        self.DeformAttn2 = MSDeformAttn(d_model=self.lidar_post_dim + self.radar_post_dim, d_v=self.radar_post_dim,
                                        n_levels=1, n_heads=8,
                                        n_points=8)  # d_model=256, n_levels=1, n_heads=8, n_points=4
        self.LearnedPositionalEncoding = LearnedPositionalEncoding3D(
            num_feats=(self.radar_post_dim + self.lidar_post_dim) // 2,
            row_num_embed=self.bev_size,
            col_num_embed=self.bev_size)
        self.radar_deconv = nn.Sequential(
            nn.Conv2d(self.radar_pre_dim, self.radar_post_dim, 1, padding=0, bias=False),
            nn.BatchNorm2d(self.radar_post_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(True)
        )

        self.lidar_deconv = nn.Sequential(
            nn.Conv2d(self.lidar_pre_dim, self.lidar_post_dim, 1, padding=0, bias=False),
            nn.BatchNorm2d(self.lidar_post_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(True)
        )

        self.RadarConvFuser_fuse = nn.Sequential(
            nn.Conv2d(self.radar_post_dim + self.lidar_post_dim, self.lidar_post_dim, 1, padding=0, bias=False),
            nn.BatchNorm2d(self.lidar_post_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(True)
        )

    def forward(self, batch_dict):
        feat_lidar = batch_dict['spatial_features_2d']
        feat_radar = batch_dict['radar_spatial_features_2d']
        # print(feat_lidar.shape)
        # print(feat_radar.shape)
        fusion_feats = []

        feat_radar = self.radar_deconv(feat_radar)
        feat_lidar = self.lidar_deconv(feat_lidar)
        radar_feats = rearrange(feat_radar, 'b c h w -> b (h w) c')
        lidar_feats = rearrange(feat_lidar, 'b c h w -> b (h w) c')

        device = torch.device("cuda")  # Get the CUDA device
        mask = torch.zeros(1, 1, self.bev_size, self.bev_size).to(device)
        pos1 = self.LearnedPositionalEncoding1(mask)
        pos2 = self.LearnedPositionalEncoding2(mask)


        reference_point1 = self.get_reference_points(self.bev_size, self.bev_size, device=device)
        reference_point2 = self.get_reference_points(self.bev_size, self.bev_size, device=device)

        fusion_f1 = self.DeformAttn1(query=self.with_pos_embed(lidar_feats, pos1),
                                     reference_points=reference_point1,
                                     input_flatten=self.with_pos_embed(lidar_feats, pos1),
                                     input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                         device),
                                     input_level_start_index=torch.tensor(
                                         [0, self.bev_size * self.bev_size]).to(device),
                                     input_padding_mask=None)
        fusion_f2 = self.DeformAttn2(query=self.with_pos_embed(lidar_feats, pos1),
                                     reference_points=reference_point2,
                                     input_flatten=self.with_pos_embed(radar_feats, pos2),
                                     input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                         device),
                                     input_level_start_index=torch.tensor(
                                         [0, self.bev_size * self.bev_size]).to(device),
                                     input_padding_mask=None)
        fusion_f1 = rearrange(fusion_f1, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
        fusion_f2 = rearrange(fusion_f2, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
        fusion_f = torch.cat((fusion_f1, fusion_f2), dim=1)
        fusion_f = self.RadarConvFuser_fuse(fusion_f)

        batch_dict['spatial_features_2d'] = fusion_f

        return batch_dict
