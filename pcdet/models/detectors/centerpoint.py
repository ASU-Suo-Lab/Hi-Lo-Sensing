from .detector3d_template import Detector3DTemplate
import torch

class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 
            'backbone_2d', 'dense_head'
        ]
        self.vis_feat = model_cfg.get('VIS_FEAT',None)

    def forward(self, batch_dict):
        for i,cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
            if self.vis_feat is not None:
              module_name = self.module_topology[i]
              frame_ids = batch_dict['frame_id']
              if module_name == self.vis_feat.MODULE:
                if module_name == 'map_to_bev_module': 
                  feature = batch_dict['spatial_features']
                elif module_name ==  'backbone_2d':
                  feature = batch_dict['spatial_features_2d']
                frame_ids = frame_ids.astype(str).tolist()
                print(f"******save {module_name} feature map******")
                for idx, frame_id in enumerate(frame_ids):
                  torch.save(feature[idx].cpu(), f'/scratch/sding32/weights/{frame_id}_{module_name}_feature.pt')

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
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
