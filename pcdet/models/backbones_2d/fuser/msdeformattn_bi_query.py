import torch
from torch import nn
from einops import rearrange
from pcdet.ops.ms_deform_attn import MSDeformAttn


class LearnedPositionalEncoding3D(nn.Module):
    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        # 行和列嵌入的特征维度
        self.num_feats = num_feats
        # 行和列嵌入的字典大小
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        # [w, num_feats]
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        # [h, w, 2*num_feats],其中每个位置的编码由行和列的嵌入向量组成,再调整为[bs, num_feats*2, h, w].
        pos = torch.cat(
            # 得到一个形状为 [h, w, num_feats] 的张量，使每一行都拥有相同的列嵌入
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        pos = rearrange(pos, 'b c h w -> b (h w) c')
        return pos

class RadarConvFuser(nn.Module):
    def __init__(self, in_channels, out_channels, deconv_blocks) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        deconv = []
        deconv_in = [sum(in_channels) + out_channels]
        deconv_out = [out_channels]
        for i in range(deconv_blocks - 1):
            deconv_in.append(out_channels)
            deconv_out.append(out_channels)
        for i in range(deconv_blocks):
            deconv.append(nn.Sequential(
                nn.Conv2d(deconv_in[i], deconv_out[i], 3, padding=1, bias=False),
                nn.BatchNorm2d(deconv_out[i]),
                nn.ReLU(True))
            )
        self.deconv = nn.ModuleList(deconv)

    def init_weights(self):
        super().init_weights()
        normal_init(self.fuse_conv, mean=0, std=0.001)
        for i in enumerate(self.deconv):
            normal_init(i, mean=0, std=0.001)

    def forward(self, input1, input2) -> torch.Tensor:
        res = torch.cat((input1, input2), dim=1)
        res2 = res.clone()
        out = self.fuse_conv(res)
        out = torch.cat([out, res2], dim=1)
        for layer in self.deconv:
            out = layer(out)
        return out


class LRMSDeformAttnFuser(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        in_channels = model_cfg.IN_CHANNEL
        out_channels = model_cfg.OUT_CHANNEL
        bev_size = model_cfg.BEV_MAP_SIZE
        self.radar_pre_dim, self.lidar_pre_dim = in_channels[0], in_channels[1]
        self.radar_post_dim, self.lidar_post_dim = out_channels[0], out_channels[1]
        self.bev_size = bev_size
        self.DeformAttn1 = MSDeformAttn(d_model=self.radar_post_dim, n_levels=1, n_heads=8, n_points=8)  # d_model=256, n_levels=1, n_heads=8, n_points=4
        self.DeformAttn2 = MSDeformAttn(d_model=self.lidar_post_dim, n_levels=1, n_heads=8, n_points=8)  # d_model=256, n_levels=1, n_heads=8, n_points=4
        self.LearnedPositionalEncoding1 = LearnedPositionalEncoding3D(num_feats=self.radar_post_dim // 2,
                                                                      row_num_embed=bev_size,
                                                                      col_num_embed=bev_size)
        self.LearnedPositionalEncoding2 = LearnedPositionalEncoding3D(num_feats=self.lidar_post_dim // 2,
                                                                      row_num_embed=bev_size,
                                                                      col_num_embed=bev_size)

        self.radar_deconv = nn.Sequential(
            nn.Conv2d(self.radar_pre_dim, self.radar_post_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.radar_post_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(True)
        )

        self.lidar_deconv = nn.Sequential(
            nn.Conv2d(self.lidar_pre_dim, self.lidar_post_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.lidar_post_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(True)
        )
        
        self.RadarConvFuser_fuse = RadarConvFuser(in_channels=(self.radar_post_dim, self.lidar_post_dim),
                                                  out_channels=self.lidar_post_dim,
                                                  deconv_blocks=3)
        
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='2d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, batch_dict):
        feat_lidar =  batch_dict['spatial_features_2d']
        feat_radar =  batch_dict['radar_spatial_features_2d']
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

        fusion_f1 = self.DeformAttn1(query=self.with_pos_embed(radar_feats, pos1),
                                         reference_points=reference_point1,
                                         input_flatten=self.with_pos_embed(lidar_feats, pos2),
                                         input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                         input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                         input_padding_mask=None)
        fusion_f2 = self.DeformAttn2(query=self.with_pos_embed(lidar_feats, pos2),
                                         reference_points=reference_point2,
                                         input_flatten=self.with_pos_embed(radar_feats, pos1),
                                         input_spatial_shapes=torch.tensor([(self.bev_size, self.bev_size)]).to(
                                                 device),
                                         input_level_start_index=torch.tensor(
                                                 [0, self.bev_size * self.bev_size]).to(device),
                                         input_padding_mask=None)
        fusion_f1 = rearrange(fusion_f1, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
        fusion_f2 = rearrange(fusion_f2, 'b (h w) c -> b c h w', h=self.bev_size, w=self.bev_size)
        fusion_f = self.RadarConvFuser_fuse(fusion_f1, fusion_f2)

        batch_dict['spatial_features_2d'] = fusion_f

        return batch_dict
