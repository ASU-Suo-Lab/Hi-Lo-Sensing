import os
import math

import torch
import torch.distributed as dist
import tqdm
import time
import glob
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils


def _ddp_all_ranks_true(flag: bool, device: torch.device) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return flag
    flag_tensor = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag_tensor, op=dist.ReduceOp.MIN)
    return bool(flag_tensor.item())


def _optimizer_zero_grad(optimizer) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _collect_nonfinite_grad_names(model, limit: int = 6):
    bad_names = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad_names.append(name)
            if len(bad_names) >= limit:
                break
    return bad_names


def _collect_large_grad_names(model, limit: int = 6):
    grad_stats = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_abs_max = float(param.grad.detach().abs().max().item())
        if math.isfinite(grad_abs_max):
            grad_stats.append((grad_abs_max, name))
    grad_stats.sort(reverse=True)
    return [name for _, name in grad_stats[:limit]]


def _sanitize_nonfinite_gradients(model, clamp_value: float = 1e3) -> bool:
    found = False
    for _, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            found = True
            param.grad.data = torch.nan_to_num(
                param.grad.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value
            )
        if clamp_value is not None:
            param.grad.data.clamp_(min=-clamp_value, max=clamp_value)
    return found


def _clip_grad_values_(model, clip_value: float) -> None:
    if clip_value is None:
        return
    for _, param in model.named_parameters():
        if param.grad is None:
            continue
        param.grad.data.clamp_(min=-clip_value, max=clip_value)


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, 
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None, 
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False,
                    use_amp=False, amp_dtype='fp16'):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    autocast_dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16
    use_grad_scaler = use_amp and amp_dtype == 'fp16'
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))
    grad_value_clip = optim_cfg.get('GRAD_VALUE_CLIP', None)
    
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()

    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter, cur_epoch)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        _optimizer_zero_grad(optimizer)

        with torch.amp.autocast('cuda', enabled=use_amp, dtype=autocast_dtype):
            loss, tb_dict, disp_dict = model_func(model, batch)

        loss_is_finite = _ddp_all_ranks_true(bool(torch.isfinite(loss).all().item()), loss.device)
        if not loss_is_finite:
            _optimizer_zero_grad(optimizer)
            if rank == 0 and logger is not None:
                logger.warning('Skip iteration %d due to non-finite loss.', accumulated_iter)
            accumulated_iter += 1
            end = time.time()
            continue

        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _clip_grad_values_(model, grad_value_clip)
            grad_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            grad_is_finite = _ddp_all_ranks_true(bool(torch.isfinite(grad_norm).all().item()), loss.device)
            if grad_is_finite:
                scaler.step(optimizer)
                scaler.update()
            else:
                bad_names = _collect_nonfinite_grad_names(model)
                if not bad_names:
                    bad_names = _collect_large_grad_names(model)
                sanitized = _sanitize_nonfinite_gradients(model, clamp_value=grad_value_clip or 1e3)
                _clip_grad_values_(model, grad_value_clip)
                if sanitized or bad_names:
                    grad_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                    grad_is_finite = _ddp_all_ranks_true(bool(torch.isfinite(grad_norm).all().item()), loss.device)
                if grad_is_finite:
                    if rank == 0 and logger is not None:
                        logger.warning(
                            'Sanitized non-finite gradients at iteration %d. Example params: %s',
                            accumulated_iter, ', '.join(bad_names) if bad_names else '<unknown>'
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    _optimizer_zero_grad(optimizer)
                    if rank == 0 and logger is not None:
                        logger.warning(
                            'Skip iteration %d due to non-finite gradients. Example params: %s',
                            accumulated_iter, ', '.join(bad_names) if bad_names else '<unknown>'
                        )
        else:
            loss.backward()
            _clip_grad_values_(model, grad_value_clip)
            grad_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            grad_is_finite = _ddp_all_ranks_true(bool(torch.isfinite(grad_norm).all().item()), loss.device)
            if grad_is_finite:
                optimizer.step()
            else:
                bad_names = _collect_nonfinite_grad_names(model)
                if not bad_names:
                    bad_names = _collect_large_grad_names(model)
                sanitized = _sanitize_nonfinite_gradients(model, clamp_value=grad_value_clip or 1e3)
                _clip_grad_values_(model, grad_value_clip)
                if sanitized or bad_names:
                    grad_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                    grad_is_finite = _ddp_all_ranks_true(bool(torch.isfinite(grad_norm).all().item()), loss.device)
                if grad_is_finite:
                    if rank == 0 and logger is not None:
                        logger.warning(
                            'Sanitized non-finite gradients at iteration %d. Example params: %s',
                            accumulated_iter, ', '.join(bad_names) if bad_names else '<unknown>'
                        )
                    optimizer.step()
                else:
                    _optimizer_zero_grad(optimizer)
                    if rank == 0 and logger is not None:
                        logger.warning(
                            'Skip iteration %d due to non-finite gradients. Example params: %s',
                            accumulated_iter, ', '.join(bad_names) if bad_names else '<unknown>'
                        )

        accumulated_iter += 1
 
        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            batch_size = batch.get('batch_size', None)
            
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item() , batch_size)
            
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })
            
            if use_logger_to_record:
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)
                    
                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} ' 
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch+1,total_epochs, 100. * (cur_epoch+1) / total_epochs,
                            cur_it,total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                            )
                    )
                    
                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:                
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
            
            # save intermediate ckpt every {ckpt_save_time_interval} seconds         
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1
                
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_amp=False, amp_dtype='fp16',
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None, show_gpu_stat=False, cfg=None):
    accumulated_iter = start_iter

    # use for disable data augmentation hook
    hook_config = cfg.get('HOOK', None) 
    augment_disable_flag = False

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            
            augment_disable_flag = disable_augmentation_hook(hook_config, dataloader_iter, total_epochs, cur_epoch, cfg, augment_disable_flag, logger)
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, 
                
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record, 
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval, 
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp,
                amp_dtype=amp_dtype
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)


def disable_augmentation_hook(hook_config, dataloader, total_epochs, cur_epoch, cfg, flag, logger):
    """
    This hook turns off the data augmentation during training.
    """
    if hook_config is not None:
        DisableAugmentationHook = hook_config.get('DisableAugmentationHook', None)
        if DisableAugmentationHook is not None:
            num_last_epochs = DisableAugmentationHook.NUM_LAST_EPOCHS
            if (total_epochs - num_last_epochs) <= cur_epoch and not flag:
                DISABLE_AUG_LIST = DisableAugmentationHook.DISABLE_AUG_LIST
                dataset_cfg=cfg.DATA_CONFIG
                logger.info(f'Disable augmentations: {DISABLE_AUG_LIST}')
                dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = DISABLE_AUG_LIST
                dataloader._dataset.data_augmentor.disable_augmentation(dataset_cfg.DATA_AUGMENTOR)
                flag = True
    return flag
