import os
import pdb
import sys
import pwd
import torch
import random 
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import setup_seed, carla_do_eval, evaluate_one_epoch, train_one_epoch, setup_configurations, print_best_results, update_best_and_save, read_pickle, write_pickle, write_json
from dataset import Carla, get_dataloaders, get_carla_dataset
from loss import Loss

import setproctitle

# Rename the process to an anonymous name
setproctitle.setproctitle("")


def main(args):
    
    best_results, now, train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader, device, CLASSES, LABEL2CLASSES, pointpillars, loss_func, optimizer, scheduler, writer, saved_ckpt_path, saved_print_path = \
        setup_configurations(args, setup_seed, get_carla_dataset, get_dataloaders, Loss, Carla)
    
    if len(args.log_name) > 0:
        f = open(os.path.join(saved_print_path, 'eval_results.txt'), 'a')
    else:
        f = sys.stdout

    print(f'==================args==================\n{args}', file=f)
    all_results = {}

    # Main loop
    for epoch in range(args.max_epoch):
        print(f'==================Epoch {epoch}================== | {args.log_name} | {args.data_root}')
        train_one_epoch(epoch, train_dataloader, pointpillars, optimizer, scheduler, loss_func, writer, device, args, f=f)
        
        val_results = evaluate_one_epoch(epoch, val_dataloader, pointpillars, val_dataset, saved_print_path, args, device, CLASSES, LABEL2CLASSES, carla_do_eval, mode='val', f=f)
        test_results = evaluate_one_epoch(epoch, test_dataloader, pointpillars, test_dataset, saved_print_path, args, device, CLASSES, LABEL2CLASSES, carla_do_eval, mode='test', f=f)
        best_results = update_best_and_save(pointpillars, val_results, test_results, best_results, saved_ckpt_path, epoch, f)

        all_results[epoch] = {
            'val': val_results,
            'test': test_results
        }

        f.flush()

    print(f'==================Final==================', file=f)
    print_best_results(best_results, f, epoch)
    print(f'all_results: \n{all_results}\n\n')
    print(f'best_results: \n{best_results}\n\n')
    
    write_json(best_results, os.path.join(saved_print_path, 'best_results.json'))
    write_json(all_results, os.path.join(saved_print_path, 'all_results.json'))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='./data/processed/3lidar=low_3radar=high/fused', help='your data root for kitti')
    parser.add_argument('--saved_path', default='pillar_logs')
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