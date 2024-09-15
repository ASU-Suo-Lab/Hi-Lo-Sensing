import numpy as np
import os
import torch
from torch.utils.data import Dataset

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

from utils import read_pickle, read_points, bbox_camera2lidar
from dataset import point_range_filter, data_augment, single_object_filter
from dataset import carla_point_range_filter, carla_data_augment, carla_lidar_radar_point_range_filter



def get_key(val, dictionary):
    for key, value in dictionary.items():
        if value == val:
            return key
    return None

class BaseSampler():
    def __init__(self, sampled_list, shuffle=True):
        self.total_num = len(sampled_list)
        self.sampled_list = np.array(sampled_list)
        self.indices = np.arange(self.total_num)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle
        self.idx = 0

    def sample(self, num):
        if self.idx + num < self.total_num:
            ret = self.sampled_list[self.indices[self.idx:self.idx+num]]
            self.idx += num
        else:
            ret = self.sampled_list[self.indices[self.idx:]]
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
        return ret


def get_carla_dataset(data_root, pcr, prefix='carla', pts_prefix='velodyne', dim=4, n_frame=1):
    # pcr = [-37, -50, -0.1, 40, 29, 3.9]
    # pcr = [-18, -24, -0.1, 22, 7, 3.9]
    CLASSES = {
        'Car': 0,
        'Pedestrian': 1, 
        'Cyclist': 2
    }

    data_infos = read_pickle(os.path.join(data_root, f'carla_infos.pkl'))["data_list"]
    
    filtered_data_infos = []
    for data_info in data_infos:
        instances = []
        annos = {
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            'score': [],
            'difficulty': []
        }
        for instance in data_info['instances']:
            if single_object_filter(instance['bbox_3d'], pcr):
                annos['name'].append(get_key(instance['bbox_label_3d'], CLASSES))
                annos['truncated'].append(instance['truncated'])
                annos['occluded'].append(instance['occluded'])
                annos['alpha'].append(instance['alpha'])
                annos['bbox'].append(instance['bbox'])
                annos['location'].append(instance['bbox_3d'][:3])
                annos['dimensions'].append(instance['bbox_3d'][3:6])
                annos['rotation_y'].append(instance['bbox_3d'][6])
                annos['score'].append(instance['score'])
                annos['difficulty'].append(instance['difficulty'])
                instances.append(instance)
        if len(instances) == 0:
            print(f'No gt_bboxes_3d: {data_info["sample_idx"]}')
        else:
            data_info['instances'] = np.array(instances)
            data_info['annos'] = {k:np.array(v) for k, v in annos.items()}
            filtered_data_infos.append(data_info)
    
    # assert False
    # self.data_infos = read_pickle(os.path.join(data_root, f'{prefix}_infos_{split}.pkl'))["data_list"]
    filtered_data_infos = sorted(filtered_data_infos, key=lambda x: int(x['sample_idx']))
    data_infos = filtered_data_infos

    # train: val: test = 0.6: 0.2: 0.2
    train_data_infos = data_infos[:int(len(data_infos) * 0.6)]
    val_data_infos = data_infos[int(len(data_infos) * 0.6):int(len(data_infos) * 0.8)]
    test_data_infos = data_infos[int(len(data_infos) * 0.8):]

    train_dataset = Carla(data_root, pcr, 'train', prefix=prefix, pts_prefix=pts_prefix, dim=dim, n_frame=n_frame, data_infos=train_data_infos)
    val_dataset = Carla(data_root, pcr, 'val', prefix=prefix, pts_prefix=pts_prefix, dim=dim, n_frame=n_frame, data_infos=val_data_infos)
    test_dataset = Carla(data_root, pcr, 'test', prefix=prefix, pts_prefix=pts_prefix, dim=dim, n_frame=n_frame, data_infos=test_data_infos)

    return train_dataset, val_dataset, test_dataset


class Carla(Dataset):

    CLASSES = {
        'Car': 0,
        'Pedestrian': 1, 
        'Cyclist': 2
        }

    def __init__(self, data_root, pcr, split, prefix='carla', pts_prefix='velodyne', dim=4, n_frame=1, data_infos=None):
        assert split in ['train', 'val', 'trainval', 'test']
        self.pcr = pcr
        self.dim = dim
        self.n_frame = n_frame
        self.data_root = data_root
        self.split = split
        self.pts_prefix = pts_prefix
        if data_infos is None:
            data_infos = read_pickle(os.path.join(data_root, f'{prefix}_infos_{split}.pkl'))["data_list"]
            
            filtered_data_infos = []
            for data_info in data_infos:
                instances = []
                annos = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': [],
                    'difficulty': []
                }
                for instance in data_info['instances']:
                    if single_object_filter(instance['bbox_3d'], self.pcr):
                        annos['name'].append(get_key(instance['bbox_label_3d'], self.CLASSES))
                        annos['truncated'].append(instance['truncated'])
                        annos['occluded'].append(instance['occluded'])
                        annos['alpha'].append(instance['alpha'])
                        annos['bbox'].append(instance['bbox'])
                        annos['location'].append(instance['bbox_3d'][:3])
                        annos['dimensions'].append(instance['bbox_3d'][3:6])
                        annos['rotation_y'].append(instance['bbox_3d'][6])
                        annos['score'].append(instance['score'])
                        annos['difficulty'].append(instance['difficulty'])
                        # print(f'difficulty: {instance["difficulty"]}')
                        instances.append(instance)
                if len(instances) == 0:
                    print(f'No gt_bboxes_3d: {data_info}')
                else:
                    data_info['instances'] = np.array(instances)
                    data_info['annos'] = {k:np.array(v) for k, v in annos.items()}
                    filtered_data_infos.append(data_info)
            
            # assert False
            # self.data_infos = read_pickle(os.path.join(data_root, f'{prefix}_infos_{split}.pkl'))["data_list"]
            filtered_data_infos = sorted(filtered_data_infos, key=lambda x: int(x['sample_idx']))
            self.data_infos = filtered_data_infos
        else:
            self.data_infos = data_infos

        # count number of agents
        self.pedestrian_count = 0
        self.cyclist_count = 0
        self.car_count = 0
        for data_info in self.data_infos:
            for name in data_info['annos']['name']:
                if name == 'Pedestrian':
                    self.pedestrian_count += 1
                elif name == 'Cyclist':
                    self.cyclist_count += 1
                elif name == 'Car':
                    self.car_count += 1
        
        self.data_aug_config=dict(
            object_noise=dict(
                num_try=100,
                translation_std=[1.0, 1.0, 0.5],
                global_rot_range=[0.0, 0.0],
                rot_range=[-0.78539816, 0.78539816],
                ),
            random_flip_ratio=0.5,
            global_rot_scale_trans=dict(
                rot_range=[-0.78539816, 0.78539816],
                scale_ratio_range=[0.95, 1.05],
                translation_std=[0, 0, 0]
                ), 
            point_range_filter=self.pcr,
            object_range_filter=self.pcr             
        )

    def remove_dont_care(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info

    def filter_db(self, db_infos):
        # 1. filter_by_difficulty
        for k, v in db_infos.items():
            db_infos[k] = [item for item in v if item['difficulty'] != -1]

        # 2. filter_by_min_points, dict(Car=5, Pedestrian=10, Cyclist=10)
        filter_thrs = dict(Car=15, Pedestrian=5, Cyclist=10)
        for cat in self.CLASSES:
            filter_thr = filter_thrs[cat]
            db_infos[cat] = [item for item in db_infos[cat] if item['num_points_in_gt'] >= filter_thr]
        
        return db_infos

    def __len__(self):
        return len(self.data_infos) - self.n_frame + 1
         
    def __getitem__(self, index):
    
        start_index = index
        end_index = start_index + self.n_frame
        
        # Ensure you do not go out of bounds
        if end_index > len(self.data_infos):
            end_index = len(self.data_infos)
        
        data_infos = self.data_infos[start_index:end_index]
        data_dicts = []

        for data_info in data_infos:
            sample_idx = data_info['sample_idx']
            lidar_velodyne_path = data_info['lidar_points']['lidar_path']
            lidar_pts_path = os.path.join(self.data_root, 'velodyne', lidar_velodyne_path)
            lidar_pts = read_points(lidar_pts_path, dim=self.dim)
            
            radar_velodyne_path = data_info['radar_points']['radar_path']
            radar_pts_path = os.path.join(self.data_root, 'velodyne', radar_velodyne_path)
            radar_pts = read_points(radar_pts_path, dim=self.dim)
            
            # print(f'lidar_path: {lidar_pts_path} | radar_path: {radar_pts_path}')
            # print(f'lidar_pts: {lidar_pts.shape} | radar_pts: {radar_pts.shape}')

            if lidar_pts.shape[1] != self.dim or radar_pts.shape[1] != self.dim:
                assert False
            
            gt_bboxes_3d, gt_names, gt_labels, difficulty = [], [], [], []
            for instance in data_info['instances']:
                gt_bboxes_3d.append(instance['bbox_3d'])
                gt_names.append(get_key(instance['bbox_label_3d'], self.CLASSES))
                gt_labels.append(instance['bbox_label_3d'])
                difficulty.append(instance['difficulty'])

            gt_bboxes_3d = np.array(gt_bboxes_3d)
            gt_labels = np.array(gt_labels)
            gt_names = np.array(gt_names)
            difficulty = np.array(difficulty)
            
            data_dict = {
                'scene_id': sample_idx,
                'lidar_pts': lidar_pts,
                'radar_pts': radar_pts,
                'gt_bboxes_3d': gt_bboxes_3d,
                'gt_labels': gt_labels, 
                'gt_names': gt_names,
                'difficulty': difficulty,
            }


            data_dict = carla_lidar_radar_point_range_filter(data_dict, point_range=self.data_aug_config['point_range_filter'])

            if data_dict['radar_pts'].shape[0] in [0, 1]:# workaround for empty radar points
                data_dict['radar_pts'] = data_dict['lidar_pts'][-2:, :] 
                # print(f'scene_id: {sample_idx} | lidar_pts: {data_dict["lidar_pts"].shape} | radar_pts: {data_dict["radar_pts"].shape}')

            raw_fused_pts = np.concatenate([data_dict['lidar_pts'], data_dict['radar_pts']], axis=0)
            data_dict['fused_pts'] = raw_fused_pts

            data_dicts.append(data_dict)
        
        return data_dicts

if __name__ == '__main__':
    pass