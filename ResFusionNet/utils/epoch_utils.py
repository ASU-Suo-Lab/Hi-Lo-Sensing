import os
import glob
import torch
import numpy as np
from tqdm import tqdm



def update_best_and_save(pointpillars, val_results, test_results, best_results, saved_ckpt_path, epoch, f):

    iou_levels = ['0.25', '0.5', '0.7', 'mean']
    agent_types = ['Pedestrian', 'Cyclist', 'Car']

    for iou in iou_levels:
        for agent in agent_types:
            if iou == 'mean':
                val_metric = np.mean(val_results["bbox_3dspecifics"][agent])
                best_metric = np.mean(best_results[f"val_{iou}"][agent])
            else:
                index = {'0.25': -1, '0.5': -2, '0.7': -3}[iou]
                val_metric = val_results["bbox_3dspecifics"][agent][index]
                best_metric = best_results[f"val_{iou}"][agent][index]

            if val_metric >= best_metric:
                best_results[f"val_{iou}"][agent] = val_results["bbox_3dspecifics"][agent]
                best_results[f"test_{iou}"][agent] = test_results["bbox_3dspecifics"][agent]
                print_best_results(best_results, f, epoch, iou=iou, agent_type=agent)
                if saved_ckpt_path is not None:
                    torch.save(pointpillars.state_dict(), os.path.join(saved_ckpt_path, f'{agent[:3]}_{iou}_best_epoch_{epoch}.pth'))

                    # Remove the previous best model
                    previous_best_models = glob.glob(os.path.join(saved_ckpt_path, f'{agent[:3]}_{iou}_best_epoch_*.pth'))
                    for model_path in previous_best_models:
                        if model_path != os.path.join(saved_ckpt_path, f'{agent[:3]}_{iou}_best_epoch_{epoch}.pth'):
                            os.remove(model_path)

    return best_results


def print_best_results(best_results, f, epoch, iou=None, agent_type=None):

    if iou is None:
        iou_list = ['0.25', '0.5', '0.7', 'mean']
    else:
        iou_list = [iou]

    for iou in iou_list:
        print(f'==================Epoch_{epoch} | {agent_type} | Best IOU_{iou} | VAL results==================', file=f)
        for k, v in best_results[f"val_{iou}"].items():
            if k == 'Cyclist':
                k = 'Cyclist   '
            elif k == 'Car':
                k = 'Car       '
            print(f'{k} AP@IOU(0.70, 0.50, 0.25): {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)
        print(f'==================Epoch_{epoch} | {agent_type} | Best IOU_{iou} | TEST results==================', file=f)
        for k, v in best_results[f"test_{iou}"].items():
            if k == 'Cyclist':
                k = 'Cyclist   '
            elif k == 'Car':
                k = 'Car       '
            print(f'{k} AP@IOU(0.70, 0.50, 0.25): {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}', file=f)      
        print(f'\n\n', file=f)

    print(f'\n\n\n', file=f)



def save_summary(writer, loss_dict, global_step, tag, lr=None, momentum=None):
    for k, v in loss_dict.items():
        writer.add_scalar(f'{tag}/{k}', v, global_step)
    if lr is not None:
        writer.add_scalar('lr', lr, global_step)
    if momentum is not None:
        writer.add_scalar('momentum', momentum, global_step)




def train_one_epoch(epoch, train_dataloader, pointpillars, optimizer, scheduler, loss_func, writer, device, args, f):
    # print('=' * 20, epoch, '=' * 20, file=f)
    train_step = 0

    for data_dict in tqdm(train_dataloader):
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
        
        optimizer.zero_grad()

        batched_lidar_pts = data_dict['batched_lidar_pts']
        batched_radar_pts = data_dict['batched_radar_pts']
        batched_fused_pts = data_dict['batched_fused_pts']
        batched_gt_bboxes = data_dict['batched_gt_bboxes']
        batched_labels = data_dict['batched_labels']
        
        # print(type(batched_lidar_pts), type(batched_radar_pts), type(batched_fused_pts), type(batched_gt_bboxes), type(batched_labels))
        # print(len(batched_lidar_pts), len(batched_radar_pts), len(batched_fused_pts), len(batched_gt_bboxes), len(batched_labels))
        # print(len(batched_lidar_pts[0]), len(batched_radar_pts[0]), len(batched_fused_pts[0]), len(batched_gt_bboxes[0]), len(batched_labels[0]))
        # print(batched_lidar_pts[0][-1].shape, batched_radar_pts[0][-1].shape, batched_fused_pts[0][-1].shape, batched_gt_bboxes[0].shape, batched_labels[0].shape)
        
        # assert False
        if args.fusion != 'raw':
            batched_fused_pts = None
        
        bbox_cls_pred, bbox_pred, bbox_dir_cls_pred, anchor_target_dict = pointpillars(
            batched_pts=batched_lidar_pts, 
            batched_radar_pts=batched_radar_pts,
            batched_fused_pts=batched_fused_pts,
            mode='train',
            batched_gt_bboxes=batched_gt_bboxes, 
            batched_gt_labels=batched_labels
        )
        
        bbox_cls_pred = bbox_cls_pred.permute(0, 2, 3, 1).reshape(-1, args.nclasses)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 7)
        bbox_dir_cls_pred = bbox_dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)

        batched_bbox_labels = anchor_target_dict['batched_labels'].reshape(-1)
        batched_label_weights = anchor_target_dict['batched_label_weights'].reshape(-1)
        batched_bbox_reg = anchor_target_dict['batched_bbox_reg'].reshape(-1, 7)
        batched_dir_labels = anchor_target_dict['batched_dir_labels'].reshape(-1)
        
        pos_idx = (batched_bbox_labels >= 0) & (batched_bbox_labels < args.nclasses)
        bbox_pred = bbox_pred[pos_idx]
        batched_bbox_reg = batched_bbox_reg[pos_idx]
        bbox_pred[:, -1] = torch.sin(bbox_pred[:, -1].clone()) * torch.cos(batched_bbox_reg[:, -1].clone())
        batched_bbox_reg[:, -1] = torch.cos(bbox_pred[:, -1].clone()) * torch.sin(batched_bbox_reg[:, -1].clone())
        bbox_dir_cls_pred = bbox_dir_cls_pred[pos_idx]
        batched_dir_labels = batched_dir_labels[pos_idx]

        num_cls_pos = (batched_bbox_labels < args.nclasses).sum()
        bbox_cls_pred = bbox_cls_pred[batched_label_weights > 0]
        batched_bbox_labels[batched_bbox_labels < 0] = args.nclasses
        batched_bbox_labels = batched_bbox_labels[batched_label_weights > 0]

        loss_dict = loss_func(
            bbox_cls_pred=bbox_cls_pred,
            bbox_pred=bbox_pred,
            bbox_dir_cls_pred=bbox_dir_cls_pred,
            batched_labels=batched_bbox_labels, 
            num_cls_pos=num_cls_pos, 
            batched_bbox_reg=batched_bbox_reg, 
            batched_dir_labels=batched_dir_labels
        )
        
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()
        scheduler.step()
        

        # print each loss
        # print(f'Epoch: {epoch} | Step: {train_step} | Loss: {loss.item()} | cls_loss: {loss_dict["cls_loss"].item()} | reg_loss: {loss_dict["reg_loss"].item()} | dir_cls_loss: {loss_dict["dir_cls_loss"].item()}')
        global_step = epoch * len(train_dataloader) + train_step + 1

        if global_step % args.log_freq == 0 and len(args.log_name) > 0:
            save_summary(
                writer, loss_dict, global_step, 'train',
                lr=optimizer.param_groups[0]['lr'], 
                momentum=optimizer.param_groups[0]['betas'][0]
            )
        train_step += 1





def evaluate_one_epoch(epoch, val_dataloader, pointpillars, val_dataset, saved_print_path, args,
                       device, CLASSES, LABEL2CLASSES, carla_do_eval, mode, f):
    pointpillars.eval()
    format_results = []

    with torch.no_grad():
        for data_dict in tqdm(val_dataloader):
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
                
        overall_results = carla_do_eval(epoch, format_results, val_dataset.data_infos[args.n_frame-1:], CLASSES, saved_print_path, mode, f)    
    pointpillars.train()


    return overall_results
