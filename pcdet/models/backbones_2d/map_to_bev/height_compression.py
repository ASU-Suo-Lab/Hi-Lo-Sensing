import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_type = model_cfg.TYPE
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        if self.pc_type=='Radar':
            encoded_spconv_tensor = batch_dict['radar_encoded_spconv_tensor']
        else:
            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        if self.pc_type == 'Radar':
            batch_dict['radar_spatial_features'] = spatial_features
            batch_dict['radar_spatial_features_stride'] = batch_dict['radar_encoded_spconv_tensor_stride']
        else:
            batch_dict['spatial_features'] = spatial_features
            batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
