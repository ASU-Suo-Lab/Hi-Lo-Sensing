import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial


def collate_fn(list_data):
    batched_radar_pts_list, batched_lidar_pts_list, batched_fused_pts_list, batched_gt_bboxes_list = [], [], [], []
    batched_labels_list, batched_names_list = [], []
    batched_difficulty_list, batched_sceneid_list = [], []
    for data_dict in list_data:
        if type(data_dict) is list:
            frame_radar_pts_list, frame_lidar_pts_list, frame_fused_pts_list, frame_gt_bboxes_list = [], [], [], []
            frame_labels_list, frame_names_list = [], []
            frame_difficulty_list, frame_sceneid_list = [], []

            for frame_data_dict in data_dict:
                lidar_pts, radar_pts, fused_pts, gt_bboxes_3d = frame_data_dict['lidar_pts'], frame_data_dict['radar_pts'], frame_data_dict['fused_pts'], frame_data_dict['gt_bboxes_3d']
                gt_labels, gt_names = frame_data_dict['gt_labels'], frame_data_dict['gt_names']
                difficulty = frame_data_dict['difficulty']

                frame_lidar_pts_list.append(torch.from_numpy(lidar_pts).float())
                frame_radar_pts_list.append(torch.from_numpy(radar_pts).float())
                frame_fused_pts_list.append(torch.from_numpy(fused_pts).float())
                frame_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d).float())
                frame_labels_list.append(torch.from_numpy(gt_labels).float())
                frame_names_list.append(gt_names)
                frame_difficulty_list.append(torch.from_numpy(difficulty).float())
                frame_sceneid_list.append(frame_data_dict['scene_id'])

            # print(f'frame_sceneid_list: {frame_sceneid_list}')

            batched_lidar_pts_list.append(frame_lidar_pts_list)
            batched_radar_pts_list.append(frame_radar_pts_list)
            batched_fused_pts_list.append(frame_fused_pts_list)
            batched_gt_bboxes_list.append(frame_gt_bboxes_list[-1])
            batched_labels_list.append(frame_labels_list[-1])
            batched_names_list.append(frame_names_list[-1])
            batched_difficulty_list.append(frame_difficulty_list[-1])
            batched_sceneid_list.append(frame_sceneid_list[-1])

        # else:
        #     lidar_pts, radar_pts, gt_bboxes_3d = data_dict['lidar_pts'], data_dict['radar_pts'], data_dict['gt_bboxes_3d']
        #     gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
        #     difficulty = data_dict['difficulty']

        #     batched_lidar_pts_list.append(torch.from_numpy(lidar_pts))
        #     batched_radar_pts_list.append(torch.from_numpy(radar_pts))
        #     batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        #     batched_labels_list.append(torch.from_numpy(gt_labels))
        #     batched_names_list.append(gt_names) # List(str)
        #     batched_difficulty_list.append(torch.from_numpy(difficulty))
        #     batched_sceneid_list.append(data_dict['scene_id'])
    
    rt_data_dict = dict(
        batched_scene_ids=batched_sceneid_list,
        batched_lidar_pts=batched_lidar_pts_list,
        batched_radar_pts=batched_radar_pts_list,
        batched_fused_pts=batched_fused_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_difficulty=batched_difficulty_list,
    )

    return rt_data_dict


def carla_get_dataloader(dataset, batch_size, num_workers, shuffle=True, drop_last=False):
    collate = collate_fn
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last, 
        collate_fn=collate,
    )
    return dataloader



def get_dataloaders(dataset, batch_size, num_workers):
    dataloaders = dict()
    for split in ['train', 'val', 'test']:
        if split == 'train':
            shuffle = True
            drop_last = True
        else:
            shuffle = False
            drop_last = False

        dataloaders[split] = carla_get_dataloader(
            dataset=dataset[split],
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return dataloaders["train"], dataloaders["val"], dataloaders["test"]