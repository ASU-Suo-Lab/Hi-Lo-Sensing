import numpy as np
import torch
import torch.nn as nn

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
    
class L4DRFusion(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        input_channels = model_cfg.LR_CHANNELS
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
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        l_c_in_list = [model_cfg.L_CHANNELS, *num_filters[:-1]]
        self.l_blocks = nn.ModuleList()
        self.l_deblocks = nn.ModuleList()

        r_c_in_list = [model_cfg.R_CHANNELS, *num_filters[:-1]]
        self.r_blocks = nn.ModuleList()
        self.r_deblocks = nn.ModuleList()

        self.Conv_RG = nn.ModuleList()
        self.Conv_LG = nn.ModuleList()
        
        for idx in range(num_levels):
            cur_layers = [
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

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

            self.Conv_RG.append(nn.Sequential(
                nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3,stride=1, padding=1, bias=False),
                nn.Sigmoid()
            ))
            self.Conv_LG.append(nn.Sequential(
                nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3,stride=1, padding=1, bias=False),
                nn.Sigmoid()))
            
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.l_deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.r_deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
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
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.l_deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                    self.r_deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
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

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        radar_x = data_dict['radar_spatial_features']
        lidar_x = data_dict['spatial_features']
        
        x = torch.cat([lidar_x, radar_x], dim=1)
        ups = []
        l_ups = []
        r_ups = []
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            lidar_x = self.l_blocks[i](lidar_x)
            radar_x = self.r_blocks[i](radar_x)
            lidar_x = self.Conv_LG[i](x) * lidar_x
            radar_x = self.Conv_RG[i](x) * radar_x
            
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
                l_ups.append(self.l_deblocks[i](lidar_x))
                r_ups.append(self.r_deblocks[i](radar_x))
            else:
                ups.append(x)
                l_ups.append(lidar_x)
                r_ups.append(radar_x)

        if len(ups) > 1:
            x = torch.cat([torch.cat(x, dim=1) for x in (l_ups, r_ups, ups)], dim=1)
        elif len(ups) == 1:
            x = torch.cat((l_ups[0], r_ups[0], ups[0]), dim=1)

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        
        return data_dict
