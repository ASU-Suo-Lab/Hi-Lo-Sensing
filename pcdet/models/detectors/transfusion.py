import time

import torch

from .detector3d_template import Detector3DTemplate


class TransFusion(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module',
            'backbone_2d', 'dense_head'
        ]
        self.vis_feat = model_cfg.get('VIS_FEAT', None)

    def forward(self, batch_dict):
        enable_profile = (
            not self.training
            and batch_dict.get('_enable_module_profile', False)
            and torch.cuda.is_available()
        )
        module_times_ms = {}
        module_mem_mb = {}

        for i, cur_module in enumerate(self.module_list):
            module_name = self.module_topology[i] if i < len(self.module_topology) else cur_module.__class__.__name__

            if enable_profile:
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                start_allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()

            batch_dict = cur_module(batch_dict)

            if enable_profile:
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                peak_allocated = torch.cuda.max_memory_allocated()
                peak_increase_mb = max(peak_allocated - start_allocated, 0) / (1024.0 * 1024.0)
                module_times_ms[module_name] = elapsed_ms
                module_mem_mb[module_name] = peak_increase_mb

            if self.vis_feat is not None:
                frame_ids = batch_dict['frame_id']
                if module_name == self.vis_feat.MODULE:
                    if module_name == 'map_to_bev_module':
                        feature = batch_dict['spatial_features']
                    elif module_name == 'backbone_2d':
                        feature = batch_dict['spatial_features_2d']
                    else:
                        feature = None

                    if feature is not None:
                        frame_ids = frame_ids.astype(str).tolist()
                        print(f"******save {module_name} feature map******")
                        for idx, frame_id in enumerate(frame_ids):
                            torch.save(feature[idx].cpu(), f'/scratch/sding32/weights/{frame_id}_{module_name}_feature.pt')

        if enable_profile:
            batch_dict['_profile_time_ms'] = module_times_ms
            batch_dict['_profile_mem_mb'] = module_mem_mb
            batch_dict['_profile_batch_size'] = batch_dict['batch_size']

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict

        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        if enable_profile:
            recall_dicts['_profile_time_ms'] = module_times_ms
            recall_dicts['_profile_mem_mb'] = module_mem_mb
            recall_dicts['_profile_batch_size'] = batch_dict['batch_size']
        return pred_dicts, recall_dicts

    def get_training_loss(self, batch_dict):
        disp_dict = {}

        loss_trans, tb_dict = batch_dict['loss'], batch_dict['tb_dict']
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
