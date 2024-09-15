import os
import pdb
import sys
import torch
import random 
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import setup_seed, carla_do_eval, evaluate_one_epoch, train_one_epoch, setup_configurations, print_best_results, save_dict_to_npz
from dataset import Carla, get_dataloaders, get_carla_dataset
from loss import Loss



def main(args):
    
    best_results, now, train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader, device, CLASSES, LABEL2CLASSES, pointpillars, loss_func, optimizer, scheduler, writer, saved_ckpt_path, saved_print_path = \
        setup_configurations(args, setup_seed, get_carla_dataset, get_dataloaders, Loss, Carla)
    

    # Load the state dictionary from the .pth file
    state_dict = torch.load(args.ckpt_path)

    # Load the state dictionary into the model
    pointpillars.load_state_dict(state_dict)

    pointpillars.eval()
    pointpillars.to(device)
    format_results = []

    test_results = {}

    with torch.no_grad():
        for data_dict in tqdm(test_dataloader):
            if not args.no_cuda:
                # Move tensors to the CUDA device
                for key in data_dict:
                    for j, item in enumerate(data_dict[key]):
                        if isinstance(item, list):
                            for k in range(len(item)):
                                if torch.is_tensor(item[k]):
                                    data_dict[key][j][k] = data_dict[key][j][k].to(device)
                        elif torch.is_tensor(item):
                            data_dict[key][j] = data_dict[key][j].to(device)
            
            batched_sceneid_list = data_dict['batched_scene_ids']
            batched_lidar_pts = data_dict['batched_lidar_pts']
            batched_radar_pts = data_dict['batched_radar_pts']
            batched_fused_pts = data_dict['batched_fused_pts']
            batched_gt_bboxes = data_dict['batched_gt_bboxes']
            batched_labels = data_dict['batched_labels']
    
            if args.fusion != 'raw':
                batched_fused_pts = None

            batch_results = pointpillars(
                batched_pts=batched_lidar_pts, 
                batched_radar_pts=batched_radar_pts,
                batched_fused_pts=batched_fused_pts,
                mode='val',
                batched_gt_bboxes=batched_gt_bboxes, 
                batched_gt_labels=batched_labels
            )

            # print(f'batch_results: {len(batch_results)}\n{batch_results}')
            # print(f'batched_sceneid_list: {len(batched_sceneid_list)}\n{batched_sceneid_list}')

            for batch_id, scene_id in enumerate(batched_sceneid_list):
                test_results[scene_id] = {
                    "pred": {
                        "labels": batch_results[batch_id]["labels"],
                        "bboxes": batch_results[batch_id]["lidar_bboxes"],
                        "scores": batch_results[batch_id]["scores"]
                    },
                    "gt": {
                        "labels": batched_labels[batch_id].cpu().numpy(),
                        "bboxes": batched_gt_bboxes[batch_id].cpu().numpy(),
                        'lidar_pts': batched_lidar_pts[batch_id][-1].cpu().numpy(),
                        'radar_pts': batched_radar_pts[batch_id][-1].cpu().numpy()
                    }
                }

            
            for result in batch_results:
                format_result = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }
                
                for lidar_bboxes, labels, scores in zip(result['lidar_bboxes'], result['labels'], result['scores']):
                    format_result['name'].append(LABEL2CLASSES[labels])
                    format_result['truncated'].append(0)
                    format_result['occluded'].append(0)
                    format_result['alpha'].append(-10)
                    format_result['bbox'].append([0, 0, 100, 100])
                    format_result['dimensions'].append(lidar_bboxes[3:6])
                    format_result['location'].append(lidar_bboxes[:3])
                    format_result['rotation_y'].append(lidar_bboxes[6])
                    format_result['score'].append(scores)
                
                format_results.append({k: np.array(v) for k, v in format_result.items()})
                
        overall_results = carla_do_eval(1, format_results, test_dataset.data_infos[args.n_frame-1:], CLASSES, saved_print_path, 'test', None)   
        # print(f'overall_results: {overall_results}') 
        test_results['overall_results'] = overall_results

    # Save the results to a .npz file
    save_dict_to_npz(os.path.join("./pillar_logs/test/", 'test_results.npz'), test_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    # parser.add_argument('--data_root', default='/mnt/disk1/yihong/PointPillars/data/ff', help='your data root for kitti')
    # parser.add_argument('--data_root', default='./data/ff_try', help='your data root for kitti')
    parser.add_argument('--data_root', default='./data/processed/1lidar=high_3radar=low/fused', help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
    parser.add_argument('--ckpt_path', default='./pillar_logs/checkpoints/20240826-23:29:56_lidar/Cyc_mean_best_epoch_99.pth')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--nclasses', type=int, default=3)
    parser.add_argument('--device', type=int, default=6)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fusion', type=str, default='feature', help='raw | feature')
    parser.add_argument('--n_frame', type=int, default=1)
    parser.add_argument('--init_lr', type=float, default=0.00025)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--log_freq', type=int, default=8)
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--ckpt_freq_epoch', type=int, default=20)
    parser.add_argument('--large', action='store_true', help='')
    parser.add_argument('--res', action='store_true', help='whether to use res')
    parser.add_argument('--nn_res', action='store_true', help='whether to use res')
    parser.add_argument('--lidar_only', action='store_true', default=False, help='')
    parser.add_argument('--radar_only', action='store_true', default=False, help='')
    parser.add_argument('--separate_encoder', action='store_true', default=False, help='')
    parser.add_argument('--remove_velocity', action='store_true', default=False, help='')
    parser.add_argument('--no_cuda', action='store_true', help='whether to use cuda')

    args = parser.parse_args()

    main(args)
