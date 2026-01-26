from .detector3d_template import Detector3DTemplate
from .. import backbones_2d, backbones_3d
from ..backbones_2d import map_to_bev,fuser
from ..backbones_3d import vfe
import numpy as np
import time
import torch

def mem_scope_begin(device="cuda"):
    start_alloc = torch.cuda.memory_allocated(device)
    start_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    return start_alloc, start_reserved

def mem_scope_end(start_alloc, start_reserved, device="cuda"):
    peak = torch.cuda.max_memory_allocated(device)
    end_alloc = torch.cuda.memory_allocated(device)
    end_reserved = torch.cuda.memory_reserved(device)
    return {
        "peak_increase": peak - start_alloc,          # 该模块执行区间的峰值增量（字节）
        "delta_alloc":   end_alloc - start_alloc,     # 执行后仍占用的分配变化（字节）
        "delta_reserved": end_reserved - start_reserved  # 缓存池变化（字节）
    }
def fmt_bytes(n: int) -> str:
    # 以 1024 进制显示
    for u in ["B","KiB","MiB","GiB","TiB","PiB"]:
        if abs(n) < 1024:
            return f"{n:.2f}{u}"
        n /= 1024
    return f"{n:.2f}EiB"

class LidarRadar1(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module','backbone_2d',
            'radar_vfe', 'radar_backbone_3d', 'radar_map_to_bev_module','radar_backbone_2d',
            'fuser', 'dense_head'
        ]
        self.module_list = self.build_networks()
       
    def build_radar_vfe(self,model_info_dict):
        if self.model_cfg.get('RADAR_VFE', None) is None:
            return None, model_info_dict

        radar_vfe_module = vfe.__all__[self.model_cfg.RADAR_VFE.NAME](
            model_cfg=self.model_cfg.RADAR_VFE,
            num_point_features=self.model_cfg.RADAR_VFE.IN_CHANNEL,
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size= np.array(self.model_cfg.RADAR_VFE.get('VOXEL_SIZE', model_info_dict['voxel_size'])).astype(np.float32),
            grid_size=np.array(self.model_cfg.RADAR_VFE.get('GRID_SIZE', model_info_dict['grid_size'])).astype(np.float32),
        )
        model_info_dict['module_list'].append(radar_vfe_module)

        return radar_vfe_module, model_info_dict
    
    def build_radar_backbone_3d(self,model_info_dict):
        if self.model_cfg.get('RADAR_BACKBONE_3D', None) is None:
            return None, model_info_dict
        radar_backbone_3d_module = backbones_3d.__all__[self.model_cfg.RADAR_BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.RADAR_BACKBONE_3D,
            grid_size=np.array(self.model_cfg.RADAR_BACKBONE_3D.get('GRID_SIZE', model_info_dict['grid_size'])).astype(np.int32)
        )
        model_info_dict['module_list'].append(radar_backbone_3d_module)

        return radar_backbone_3d_module, model_info_dict

    def build_radar_map_to_bev_module(self,model_info_dict):
        if self.model_cfg.get('RADAR_MAP_TO_BEV', None) is None:
            return None, model_info_dict
        radar_map_to_bev_module = map_to_bev.__all__[self.model_cfg.RADAR_MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.RADAR_MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(radar_map_to_bev_module)

        return radar_map_to_bev_module, model_info_dict

    def build_radar_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('RADAR_BACKBONE_2D', None) is None:
            return None, model_info_dict

        radar_backbone_2d_module = backbones_2d.__all__[self.model_cfg.RADAR_BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.RADAR_BACKBONE_2D,
            input_channels=self.model_cfg.RADAR_MAP_TO_BEV.NUM_BEV_FEATURES
        )
        model_info_dict['module_list'].append(radar_backbone_2d_module)
        
        return radar_backbone_2d_module, model_info_dict
    
    def build_fuser(self, model_info_dict):
        if self.model_cfg.get('FUSER', None) is None:
            return None, model_info_dict
    
        fuser_module = fuser.__all__[self.model_cfg.FUSER.NAME](
            model_cfg=self.model_cfg.FUSER
        )
        model_info_dict['module_list'].append(fuser_module)
        model_info_dict['num_bev_features'] = self.model_cfg.DENSE_HEAD.IN_CHANNEL
        return fuser_module, model_info_dict


    def forward(self, batch_dict):
        module_times_ms = {}
        module_mem = {}
        for i,cur_module in enumerate(self.module_list):
            name = getattr(cur_module, 'name', f'{i}_{cur_module.__class__.__name__}')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            s_alloc, s_reserved = mem_scope_begin()
            batch_dict = cur_module(batch_dict)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            module_mem[name] = mem_scope_end(s_alloc, s_reserved)
            module_times_ms[name] = dt_ms

        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # 友好打印（时间按 ms，显存按 KiB/MiB/GiB）
            #print("\n[Per-module latency]")
            #print({k: f"{v:.2f} ms" for k, v in module_times_ms.items()})

            print("\n[Per-module memory]")
            pretty_mem = {
                k: {
                    "peak+": fmt_bytes(v["peak_increase"]),
                    "Δalloc": fmt_bytes(v["delta_alloc"]),
                    "Δreserved": fmt_bytes(v["delta_reserved"]),
                } for k, v in module_mem.items()
            }
            print(pretty_mem)
            return pred_dicts, recall_dicts

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
