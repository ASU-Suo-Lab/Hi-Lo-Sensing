import numpy as np
import pdb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.anchors import Anchors, anchor_target, anchors2bboxes
from model.attention import SelfAttentionBlock, CrossAttentionBlock
from ops import Voxelization, nms_cuda
from utils import limit_period


class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

    @torch.no_grad()
    def forward(self, batched_pts):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), 
               coors_batch: (p1 + p2 + ... + pb, 1 + 3), 
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts) 
            # voxels_out: (num_voxels, num_points, num_features), coors_out: (num_voxels, 3)
            # num_points_per_voxel_out: (num_voxels, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)

        # (sum_voxels, num_points, num_features), (sum_voxels, 1+3), (sum_voxels, )
        return pillars, coors_batch, npoints_per_pillar


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel, x_l=None, y_l=None):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        if x_l is not None:
            self.x_l = x_l
        else:
            self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
            
        if y_l is not None:
            self.y_l = y_l
        else:        
            self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False) # the same to nn.linear
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        offset_pt_center = pillars[:, :, :3] - torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, num_points, 3)

        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)
        
        # 3. encoder
        features = torch.cat([pillars, offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp
        # In consitent with mmdet3d. 
        # The reason can be referenced to https://github.com/open-mmlab/mmdetection3d/issues/1150

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]
        

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 1, 0).contiguous() # (features, y_l, x_l)
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0) # (bs, in_channel, self.y_l, self.x_l)
        return batched_canvas



class TemporalFusion(nn.Module):
    def __init__(self, in_channels, out_channels, num_history=5):
        super(TemporalFusion, self).__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3],
            padding=[1, 1, 1],
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=[3, 3, 3],
            padding=[1, 1, 1],
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=[num_history, 3, 3],
            padding=[0, 1, 1],
            bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels)
        
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        
        # Remove the temporal dimension (squeeze)
        x = x.squeeze(2)  # Shape should now be (batch, channel, w, h)
        
        return x


class Backbone(nn.Module):
    def __init__(self, in_channel, out_channels, layer_nums, layer_strides=[2, 2, 2]):
        super().__init__()
        assert len(out_channels) == len(layer_nums)
        assert len(out_channels) == len(layer_strides)
        
        # in_channel=64, out_channels=[64, 128, 256], layer_nums=[3, 5, 5]
        
        self.multi_blocks = nn.ModuleList()
        for i in range(len(layer_strides)):
            blocks = []
            blocks.append(nn.Conv2d(in_channel, out_channels[i], 3, stride=layer_strides[i], bias=False, padding=1))
            blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            blocks.append(nn.ReLU(inplace=True))

            for _ in range(layer_nums[i]): # [3, 5, 5]
                blocks.append(nn.Conv2d(out_channels[i], out_channels[i], 3, bias=False, padding=1))
                blocks.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
                blocks.append(nn.ReLU(inplace=True))

            in_channel = out_channels[i]
            self.multi_blocks.append(nn.Sequential(*blocks))

        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: (b, c, y_l, x_l). Default: (6, 64, 496, 432)
        return: list[]. Default: [(6, 64, 248, 216), (6, 128, 124, 108), (6, 256, 62, 54)]
        '''
        outs = []
        for i in range(len(self.multi_blocks)):
            x = self.multi_blocks[i](x)
            outs.append(x)
        return outs


class Neck(nn.Module):
    def __init__(self, in_channels, upsample_strides, out_channels):
        super().__init__()
        assert len(in_channels) == len(upsample_strides)
        assert len(upsample_strides) == len(out_channels)

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(in_channels)):
            decoder_block = []
            decoder_block.append(nn.ConvTranspose2d(in_channels[i], 
                                                    out_channels[i], 
                                                    upsample_strides[i], 
                                                    stride=upsample_strides[i],
                                                    bias=False))
            decoder_block.append(nn.BatchNorm2d(out_channels[i], eps=1e-3, momentum=0.01))
            decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))
        
        # in consitent with mmdet3d
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        '''
        x: [(bs, 64, 248, 216), (bs, 128, 124, 108), (bs, 256, 62, 54)]
        return: (bs, 384, 248, 216)
        '''
        outs = []
        for i in range(len(self.decoder_blocks)):
            xi = self.decoder_blocks[i](x[i]) # (bs, 128, 248, 216)
            outs.append(xi)
        out = torch.cat(outs, dim=1)
        return out


class Head(nn.Module):
    def __init__(self, in_channel, n_anchors, n_classes):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channel, n_anchors*n_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, n_anchors*7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, n_anchors*2, 1)

        # in consitent with mmdet3d
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0)
                conv_layer_id += 1

    def forward(self, x):
        '''
        x: (bs, 384, 248, 216)
        return: 
              bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
              bbox_pred: (bs, n_anchors*7, 248, 216)
              bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        '''
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


class CarlaPointPillarsMultiFrame(nn.Module):
    def __init__(self,
                 args,
                 x_l,
                 y_l,
                 anchor_range,
                 voxel_size,
                 point_cloud_range, # Intersection
                 nclasses=3, 
                 max_num_points=32,
                 max_voxels=(16000, 40000),
                 n_frame=5):
        super().__init__()
        self.args = args
        self.n_frame = n_frame
        self.nclasses = nclasses
        self.pillar_layer = PillarLayer(voxel_size=voxel_size, 
                                        point_cloud_range=point_cloud_range, 
                                        max_num_points=max_num_points, 
                                        max_voxels=max_voxels)
        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size, 
                                            point_cloud_range=point_cloud_range, 
                                            in_channel=9, 
                                            out_channel=64,
                                            x_l=x_l, y_l=y_l) # after this we got pseude-image with shape (batch_size, out_channel, y_l, x_l)
        self.backbone = Backbone(in_channel=64, 
                                 out_channels=[64, 128, 256], 
                                 layer_nums=[3, 5, 5])
        self.neck = Neck(in_channels=[64, 128, 256], 
                         upsample_strides=[1, 2, 4], 
                         out_channels=[128, 128, 128])

        if args.separate_encoder:
                
            self.raw_pillar_layer = PillarLayer(voxel_size=voxel_size, 
                                                point_cloud_range=point_cloud_range, 
                                                max_num_points=max_num_points, 
                                                max_voxels=max_voxels)
            self.raw_pillar_encoder = PillarEncoder(voxel_size=voxel_size, 
                                                    point_cloud_range=point_cloud_range, 
                                                    in_channel=9, 
                                                    out_channel=64,
                                                    x_l=x_l, y_l=y_l) # after this we got pseude-image with shape (batch_size, out_channel, y_l, x_l)
            self.raw_backbone = Backbone(in_channel=64, 
                                         out_channels=[64, 128, 256], 
                                         layer_nums=[3, 5, 5])
            self.raw_neck = Neck(in_channels=[64, 128, 256], 
                                 upsample_strides=[1, 2, 4], 
                                 out_channels=[128, 128, 128])

        
        if args.fusion == 'feature':
            self.temporal_fusion = TemporalFusion(in_channels=768, out_channels=384, num_history=n_frame)
            if args.nn_res:
                    
                self.res_fusion = nn.Linear(max(x_l, y_l), int(max(x_l, y_l)/2))
        else:
            if self.n_frame > 1:
                self.temporal_fusion = TemporalFusion(in_channels=384, out_channels=384, num_history=n_frame)

        self.head = Head(in_channel=384, n_anchors=2*nclasses, n_classes=nclasses)
        
        # anchors
        ranges=[anchor_range, anchor_range, anchor_range]
        sizes=[[3.9, 1.6, 1.56], # car
               [0.5, 0.5, 1.73], # pedestrian
               [1.76, 0.6, 1.73]]
        rotations=[0, 1.57]
        self.anchors_generator = Anchors(ranges=ranges, 
                                         sizes=sizes, 
                                         rotations=rotations)
        
        # train
        self.assigners = [
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.5, 'neg_iou_thr': 0.35, 'min_iou_thr': 0.35},
            {'pos_iou_thr': 0.6, 'neg_iou_thr': 0.45, 'min_iou_thr': 0.45},
        ]

        # val and test
        self.nms_pre = 100
        self.nms_thr = 0.01
        self.score_thr = 0.1
        self.max_num = 50

    def get_predicted_bboxes_single(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchors):
        '''
        bbox_cls_pred: (n_anchors*3, 248, 216) 
        bbox_pred: (n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (n_anchors*2, 248, 216)
        anchors: (y_l, x_l, 3, 2, 7)
        return: 
            bboxes: (k, 7)
            labels: (k, )
            scores: (k, ) 
        '''
        # 0. pre-process 
        bbox_cls_pred = bbox_cls_pred.permute(1, 2, 0).reshape(-1, self.nclasses)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        anchors = anchors.reshape(-1, 7)
        
        bbox_cls_pred = torch.sigmoid(bbox_cls_pred)
        bbox_dir_cls_pred = torch.max(bbox_dir_cls_pred, dim=1)[1]

        # 1. obtain self.nms_pre bboxes based on scores
        inds = bbox_cls_pred.max(1)[0].topk(self.nms_pre)[1]
        bbox_cls_pred = bbox_cls_pred[inds]
        bbox_pred = bbox_pred[inds]
        bbox_dir_cls_pred = bbox_dir_cls_pred[inds]
        anchors = anchors[inds]

        # 2. decode predicted offsets to bboxes
        bbox_pred = anchors2bboxes(anchors, bbox_pred)

        # 3. nms
        bbox_pred2d_xy = bbox_pred[:, [0, 1]]
        bbox_pred2d_lw = bbox_pred[:, [3, 4]]
        bbox_pred2d = torch.cat([bbox_pred2d_xy - bbox_pred2d_lw / 2,
                                 bbox_pred2d_xy + bbox_pred2d_lw / 2,
                                 bbox_pred[:, 6:]], dim=-1) # (n_anchors, 5)

        ret_bboxes, ret_labels, ret_scores = [], [], []
        for i in range(self.nclasses):
            # 3.1 filter bboxes with scores below self.score_thr
            cur_bbox_cls_pred = bbox_cls_pred[:, i]
            score_inds = cur_bbox_cls_pred > self.score_thr
            if score_inds.sum() == 0:
                continue

            cur_bbox_cls_pred = cur_bbox_cls_pred[score_inds]
            cur_bbox_pred2d = bbox_pred2d[score_inds]
            cur_bbox_pred = bbox_pred[score_inds]
            cur_bbox_dir_cls_pred = bbox_dir_cls_pred[score_inds]
            
            # 3.2 nms core
            keep_inds = nms_cuda(boxes=cur_bbox_pred2d, 
                                 scores=cur_bbox_cls_pred, 
                                 thresh=self.nms_thr, 
                                 pre_maxsize=None, 
                                 post_max_size=None)

            cur_bbox_cls_pred = cur_bbox_cls_pred[keep_inds]
            cur_bbox_pred = cur_bbox_pred[keep_inds]
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred[keep_inds]
            cur_bbox_pred[:, -1] = limit_period(cur_bbox_pred[:, -1].detach().cpu(), 1, np.pi).to(cur_bbox_pred) # [-pi, 0]
            cur_bbox_pred[:, -1] += (1 - cur_bbox_dir_cls_pred) * np.pi

            ret_bboxes.append(cur_bbox_pred)
            ret_labels.append(torch.zeros_like(cur_bbox_pred[:, 0], dtype=torch.long) + i)
            ret_scores.append(cur_bbox_cls_pred)

        # 4. filter some bboxes if bboxes number is above self.max_num
        if len(ret_bboxes) == 0:
            return {
                'lidar_bboxes': np.array([]),
                'labels': np.array([]),
                'scores': np.array([])
            }
        ret_bboxes = torch.cat(ret_bboxes, 0)
        ret_labels = torch.cat(ret_labels, 0)
        ret_scores = torch.cat(ret_scores, 0)
        if ret_bboxes.size(0) > self.max_num:
            final_inds = ret_scores.topk(self.max_num)[1]
            ret_bboxes = ret_bboxes[final_inds]
            ret_labels = ret_labels[final_inds]
            ret_scores = ret_scores[final_inds]
        result = {
            'lidar_bboxes': ret_bboxes.detach().cpu().numpy(),
            'labels': ret_labels.detach().cpu().numpy(),
            'scores': ret_scores.detach().cpu().numpy()
        }

        return result


    def get_predicted_bboxes(self, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, batched_anchors):
        '''
        bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        bbox_pred: (bs, n_anchors*7, 248, 216)
        bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        batched_anchors: (bs, y_l, x_l, 3, 2, 7)
        return: 
            bboxes: [(k1, 7), (k2, 7), ... ]
            labels: [(k1, ), (k2, ), ... ]
            scores: [(k1, ), (k2, ), ... ] 
        '''
        results = []
        bs = bbox_cls_pred.size(0)
        for i in range(bs):
            result = self.get_predicted_bboxes_single(bbox_cls_pred=bbox_cls_pred[i],
                                                      bbox_pred=bbox_pred[i], 
                                                      bbox_dir_cls_pred=bbox_dir_cls_pred[i], 
                                                      anchors=batched_anchors[i])
            results.append(result)
        return results

    def forward(self, batched_pts, batched_radar_pts=None, batched_fused_pts=None, mode='test', batched_gt_bboxes=None, batched_gt_labels=None):
        batch_size = len(batched_pts)
        n_frames = len(batched_pts[0])

        if self.args.remove_velocity:
            for i in range(batch_size):
                for j in range(self.n_frame):
                    if batched_radar_pts is not None:
                        batched_radar_pts[i][j][:, 3] = 0
                    if batched_fused_pts is not None:
                        batched_fused_pts[i][j][:, 3] = 0

        if self.args.lidar_only or self.args.radar_only: # lidar only
            if self.args.radar_only:
                batched_pts = batched_radar_pts


            if self.n_frame > 1:
                lidar_frame_features = []
                for frame_idx in range(n_frames):
                    frame_pts = [pts[frame_idx] for pts in batched_pts]
                    pillars, coors_batch, npoints_per_pillar = self.pillar_layer(frame_pts)
                    frame_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                    frame_xs = self.backbone(frame_features)
                    frame_features = self.neck(frame_xs)
                    lidar_frame_features.append(frame_features)
                
                lidar_frame_features = torch.stack(lidar_frame_features, dim=2)
                x = self.temporal_fusion(lidar_frame_features)

            else:

                batched_pts = [pts[-1] for pts in batched_pts]
                pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
                pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                xs = self.backbone(pillar_features)
                x = self.neck(xs)
            
        else:
            if batched_fused_pts is not None: # raw fusion
                
                if self.n_frame > 1:
                    raw_frame_features = []
                    for frame_idx in range(n_frames):
                        frame_pts = [pts[frame_idx] for pts in batched_pts]
                        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(frame_pts)
                        frame_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                        frame_xs = self.backbone(frame_features)
                        frame_features = self.neck(frame_xs)
                        raw_frame_features.append(frame_features)

                    raw_frame_features = torch.stack(raw_frame_features, dim=2)
                    x = self.temporal_fusion(raw_frame_features)

                else:
                    batched_pts = [pts[-1] for pts in batched_fused_pts]

                    pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)
                    pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                    xs = self.backbone(pillar_features)
                    x = self.neck(xs)
                
            else: # feature fusion

                lidar_frame_features = []
                for frame_idx in range(n_frames):
                    frame_pts = [pts[frame_idx] for pts in batched_pts]
                    pillars, coors_batch, npoints_per_pillar = self.pillar_layer(frame_pts)
                    frame_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                    frame_xs = self.backbone(frame_features)
                    frame_features = self.neck(frame_xs)
                    lidar_frame_features.append(frame_features)

                radar_frame_features = []
                for frame_idx in range(n_frames):
                    frame_radar_pts = [pts[frame_idx] for pts in batched_radar_pts]
                        
                    if self.args.separate_encoder: # separate encoder for radar
                        pillars, coors_batch, npoints_per_pillar = self.raw_pillar_layer(frame_radar_pts)
                        frame_features = self.raw_pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                        frame_xs = self.raw_backbone(frame_features)
                        frame_features = self.raw_neck(frame_xs)

                    else:
                        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(frame_radar_pts)
                        frame_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                        frame_xs = self.backbone(frame_features)
                        frame_features = self.neck(frame_xs)
                    radar_frame_features.append(frame_features)

                frame_features = []
                for i in range(len(lidar_frame_features)):
                    lidar_frame_feature = lidar_frame_features[i]
                    radar_frame_feature = radar_frame_features[i]
                    frame_feature = torch.cat([lidar_frame_feature, radar_frame_feature], dim=1)
                    frame_features.append(frame_feature)

                frame_features = torch.stack(frame_features, dim=2)
                x = self.temporal_fusion(frame_features)

                if self.args.res or self.args.nn_res:    
                    raw_fused_batched_pts = []
                    for i in range(batch_size):
                        batch_i_lidar_pts = batched_pts[i]
                        batch_i_radar_pts = batched_radar_pts[i]

                        fused_batch_i_pts = torch.cat([batch_i_lidar_pts[-1], batch_i_radar_pts[-1]], dim=0) # last frame only
                        fused_batch_i_pts[:, 3] = 0
                        raw_fused_batched_pts.append(fused_batch_i_pts)

                    pillars, coors_batch, npoints_per_pillar = self.pillar_layer(raw_fused_batched_pts)
                    pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)
                    xs = self.backbone(pillar_features)
                    raw_x = self.neck(xs)

                    if self.args.nn_res:
                        x = self.res_fusion(torch.cat([x, raw_x], dim=-1))
                    else:
                        x += raw_x # residual connection


        # bbox_cls_pred: (bs, n_anchors*3, 248, 216) 
        # bbox_pred: (bs, n_anchors*7, 248, 216)
        # bbox_dir_cls_pred: (bs, n_anchors*2, 248, 216)
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred = self.head(x)

        # anchors
        device = bbox_cls_pred.device
        feature_map_size = torch.tensor(list(bbox_cls_pred.size()[-2:]), device=device)
        anchors = self.anchors_generator.get_multi_anchors(feature_map_size)
        batched_anchors = [anchors for _ in range(batch_size)]
        
        if mode == 'train':
            anchor_target_dict = anchor_target(batched_anchors=batched_anchors, 
                                               batched_gt_bboxes=batched_gt_bboxes, 
                                               batched_gt_labels=batched_gt_labels, 
                                               assigners=self.assigners,
                                               nclasses=self.nclasses)
            
            return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict
        elif mode == 'val':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results

        elif mode == 'test':
            results = self.get_predicted_bboxes(bbox_cls_pred=bbox_cls_pred, 
                                                bbox_pred=bbox_pred, 
                                                bbox_dir_cls_pred=bbox_dir_cls_pred, 
                                                batched_anchors=batched_anchors)
            return results
        else:
            raise ValueError   
