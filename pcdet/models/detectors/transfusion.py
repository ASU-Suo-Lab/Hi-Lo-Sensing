from .detector3d_template import Detector3DTemplate
import torch
import time

def mem_scope_begin(device="cuda"):
    torch.cuda.synchronize()
    start_alloc = torch.cuda.memory_allocated(device)
    start_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    return start_alloc, start_reserved

def mem_scope_end(start_alloc, start_reserved, device="cuda"):
    torch.cuda.synchronize()
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

class TransFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 
            'backbone_2d', 'dense_head'
        ]
        self.vis_feat = model_cfg.get('VIS_FEAT',None)

    def forward(self, batch_dict):
        module_times_ms = {}
        module_mem = {}
        for i,cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
            if self.vis_feat is not None:
              module_name = self.module_topology[i]
              frame_ids = batch_dict['frame_id']
              # if module_name == self.vis_feat.MODULE and self.vis_feat.FRAME_ID in frame_ids:
              if module_name == self.vis_feat.MODULE:
                if module_name == 'map_to_bev_module': 
                  feature = batch_dict['spatial_features']
                elif module_name ==  'backbone_2d':
                  feature = batch_dict['spatial_features_2d']
                frame_ids = frame_ids.astype(str).tolist()
                #idx = frame_ids.index(self.vis_feat.FRAME_ID)
                print(f"******save {module_name} feature map******")
                for idx, frame_id in enumerate(frame_ids):
                  torch.save(feature[idx].cpu(), f'/scratch/sding32/weights/{frame_id}_{module_name}_feature.pt')
        #for i,cur_module in enumerate(self.module_list):
        #    name = getattr(cur_module, 'name', f'{i}_{cur_module.__class__.__name__}')
        #    if torch.cuda.is_available():
        #        torch.cuda.synchronize()
        #    t0 = time.perf_counter()
        #    s_alloc, s_reserved = mem_scope_begin()
        #    batch_dict = cur_module(batch_dict)
        #    if torch.cuda.is_available():
        #        torch.cuda.synchronize()
        #    dt_ms = (time.perf_counter() - t0) * 1000.0
        #    module_mem[name] = mem_scope_end(s_alloc, s_reserved)
        #    module_times_ms[name] = dt_ms

        
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

            #print("\n[Per-module memory]")
            #pretty_mem = {
            #    k: {
            #        "peak+": fmt_bytes(v["peak_increase"]),
            #        "Δalloc": fmt_bytes(v["delta_alloc"]),
            #        "Δreserved": fmt_bytes(v["delta_reserved"]),
            #    } for k, v in module_mem.items()
            #}
            #print(pretty_mem)
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
