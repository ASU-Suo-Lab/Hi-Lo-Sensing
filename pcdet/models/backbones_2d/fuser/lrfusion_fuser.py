import torch
from torch import nn
import numpy as np

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out
    
class LRFusion(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)

        l_c_in_list = [model_cfg.L_CHANNELS, *num_filters[:-1]]
        self.l_blocks = nn.ModuleList()

        r_c_in_list = [model_cfg.R_CHANNELS, *num_filters[:-1]]
        self.r_blocks = nn.ModuleList()

        self.deblocks = nn.ModuleList()
        
        for idx in range(num_levels):
            l_cur_layers = [
                BasicBlock(l_c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                l_cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.l_blocks.append(nn.Sequential(*l_cur_layers))

            r_cur_layers = [
                BasicBlock(r_c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                r_cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.r_blocks.append(nn.Sequential(*r_cur_layers))

            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            2*num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int32)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            2*num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    
        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        self.in_channels = num_filters
         # Scale 0: (B, 64, 256, 256)
        self.input_conv_lidar_scale_0 = nn.Conv2d(2 * self.in_channels[0], self.in_channels[0], kernel_size=3,
                                                  stride=1, padding=1)
        self.sigmoid_lidar_scale_0 = nn.Sigmoid()
        self.input_conv_radar_scale_0 = nn.Conv2d(2 * self.in_channels[0], self.in_channels[0], kernel_size=3,
                                                  stride=1, padding=1)
        self.sigmoid_radar_scale_0 = nn.Sigmoid()

        # Scale 1: (B, 128, 128, 128)
        self.input_conv_lidar_scale_1 = nn.Conv2d(2 * self.in_channels[1], self.in_channels[1], kernel_size=3,
                                                  stride=1, padding=1)
        self.sigmoid_lidar_scale_1 = nn.Sigmoid()
        self.input_conv_radar_scale_1 = nn.Conv2d(2 * self.in_channels[1], self.in_channels[1], kernel_size=3,
                                                  stride=1, padding=1)
        self.sigmoid_radar_scale_1 = nn.Sigmoid()
        
        # Scale 2: (B, 256, 64, 64)
        self.input_conv_lidar_scale_2 = nn.Conv2d(2 * self.in_channels[2], self.in_channels[2], kernel_size=3,
                                                  stride=1, padding=1)
        self.sigmoid_lidar_scale_2 = nn.Sigmoid()
        self.input_conv_radar_scale_2 = nn.Conv2d(2 * self.in_channels[2], self.in_channels[2], kernel_size=3,
                                                  stride=1, padding=1)
        self.sigmoid_radar_scale_2 = nn.Sigmoid()

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        radar_x = data_dict['radar_spatial_features']
        lidar_x = data_dict['spatial_features']

        middle_feat_lidar = []
        middle_feat_radar = []
        for i in range(len(self.l_blocks)):
            lidar_x = self.l_blocks[i](lidar_x)
            radar_x = self.r_blocks[i](radar_x)
            middle_feat_lidar.append(lidar_x)
            middle_feat_radar.append(radar_x)
        
        # Scale 0: (B, 64, 256, 256)
        # Learning the weighted generation network
        input_feat_cat_0 = torch.cat((middle_feat_lidar[0], middle_feat_radar[0]), dim=1)  # (B, 128, 256, 256)
        weight_lidar_0 = self.input_conv_lidar_scale_0(input_feat_cat_0)
        weight_lidar_0 = self.sigmoid_lidar_scale_0(weight_lidar_0)  #  (B, 64, 256, 256)
        weight_radar_0 = self.input_conv_radar_scale_0(input_feat_cat_0)
        weight_radar_0 = self.sigmoid_radar_scale_0(weight_radar_0)  #  (B, 64, 256, 256)

        # Element-wise product
        product_lidar_feat_0 = middle_feat_lidar[0] * weight_lidar_0  #  (B, 64, 256, 256)
        product_radar_feat_0 = middle_feat_radar[0] * weight_radar_0  #  (B, 64, 256, 256)

        # Concatenation and Convolution
        middle_feat_fused_0 = torch.cat((product_lidar_feat_0, product_radar_feat_0), dim=1)  #  (B, 128, 256, 256)

        # Scale 1: (B, 128, 128, 128)
        # Learning the weighted generation network
        input_feat_cat_1 = torch.cat((middle_feat_lidar[1], middle_feat_radar[1]), dim=1)  # (B, 256, 128, 128)
        weight_lidar_1 = self.input_conv_lidar_scale_1(input_feat_cat_1)
        weight_lidar_1 = self.sigmoid_lidar_scale_1(weight_lidar_1)  # (B, 128, 128, 128)
        weight_radar_1 = self.input_conv_radar_scale_1(input_feat_cat_1)
        weight_radar_1 = self.sigmoid_radar_scale_1(weight_radar_1)  # (B, 128, 128, 128)

        # Element-wise product
        product_lidar_feat_1 = middle_feat_lidar[1] * weight_lidar_1  # (B, 128, 128, 128)
        product_radar_feat_1 = middle_feat_radar[1] * weight_radar_1  # (B, 128, 128, 128)

        # Concatenation and Convolution
        middle_feat_fused_1 = torch.cat((product_lidar_feat_1, product_radar_feat_1), dim=1)  # (B, 256, 128, 128)

        # Scale 2: (B, 256, 64, 64)
        # Learning the weighted generation network
        input_feat_cat_2 = torch.cat((middle_feat_lidar[2], middle_feat_radar[2]), dim=1)  # (B, 512, 64, 64)
        weight_lidar_2 = self.input_conv_lidar_scale_2(input_feat_cat_2)
        weight_lidar_2 = self.sigmoid_lidar_scale_2(weight_lidar_2)  # (B, 256, 64, 64)
        weight_radar_2 = self.input_conv_radar_scale_2(input_feat_cat_2)
        weight_radar_2 = self.sigmoid_radar_scale_2(weight_radar_2)  # (B, 256, 64, 64)

        # Element-wise product
        product_lidar_feat_2 = middle_feat_lidar[2] * weight_lidar_2  # (B, 256, 64, 64)
        product_radar_feat_2 = middle_feat_radar[2] * weight_radar_2  # (B, 256, 64, 64)

        # Concatenation and Convolution
        middle_feat_fused_2 = torch.cat((product_lidar_feat_2, product_radar_feat_2), dim=1)  # (B, 512, 64, 64)

        # Final output list
        middle_feat_fused = [middle_feat_fused_0, middle_feat_fused_1, middle_feat_fused_2]

        ups = [deblock(middle_feat_fused[i]) for i, deblock in enumerate(self.deblocks)]
        
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]

        data_dict['spatial_features_2d'] = x
        
        return data_dict
