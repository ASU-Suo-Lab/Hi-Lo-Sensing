import pickle
import time

import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def _get_eval_amp_settings(cfg, args):
    use_amp = getattr(args, 'use_amp', cfg.OPTIMIZATION.get('USE_AMP', False))
    amp_dtype = getattr(args, 'amp_dtype', cfg.OPTIMIZATION.get('AMP_DTYPE', 'fp16'))
    autocast_dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16
    return use_amp, autocast_dtype


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    module_time_sums_ms = {}
    module_mem_sums_mb = {}
    profile_sample_count = 0
    pred_debug_sums = {}
    pred_debug_count = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    use_amp, autocast_dtype = _get_eval_amp_settings(cfg, args)

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    eval_start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        if getattr(args, 'infer_time', False):
            batch_dict['_enable_module_profile'] = True

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=autocast_dtype):
                pred_dicts, ret_dict = model(batch_dict)

        debug_keys = [
            'debug_raw_box_count',
            'debug_raw_score_max',
            'debug_raw_score_mean',
            'debug_post_center_count',
            'debug_post_thresh_count',
            'debug_post_filter_count',
            'debug_post_nms_count',
            'debug_post_nms_score_max',
            'debug_post_nms_score_mean',
        ]
        for pred_dict in pred_dicts:
            if any(key in pred_dict for key in debug_keys):
                pred_debug_count += 1
                for key in debug_keys:
                    if key in pred_dict:
                        pred_debug_sums[key] = pred_debug_sums.get(key, 0.0) + float(pred_dict[key])

        disp_dict = {}
        statistics_info(cfg, ret_dict, metric, disp_dict)

        if getattr(args, 'infer_time', False):
            profile_time_ms = batch_dict.get('_profile_time_ms', ret_dict.get('_profile_time_ms'))
            profile_mem_mb = batch_dict.get('_profile_mem_mb', ret_dict.get('_profile_mem_mb'))
            profile_batch_size = batch_dict.get('_profile_batch_size', ret_dict.get('_profile_batch_size', batch_dict['batch_size']))

            if profile_time_ms is not None and profile_mem_mb is not None:
                profile_sample_count += int(profile_batch_size)
                for module_name, module_time_ms in profile_time_ms.items():
                    module_time_sums_ms[module_name] = module_time_sums_ms.get(module_name, 0.0) + module_time_ms
                for module_name, module_mem_mb in profile_mem_mb.items():
                    module_mem_sums_mb[module_name] = module_mem_sums_mb.get(module_name, 0.0) + module_mem_mb

        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - eval_start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    if getattr(args, 'infer_time', False):
        if profile_sample_count > 0:
            logger.info('Average module profiling per sample:')
            for module_name, total_time_ms in module_time_sums_ms.items():
                avg_time_ms = total_time_ms / profile_sample_count
                avg_mem_mb = module_mem_sums_mb.get(module_name, 0.0) / profile_sample_count
                logger.info('%s: avg_time=%.4f ms, avg_peak_mem=%.4f MiB' % (
                    module_name, avg_time_ms, avg_mem_mb
                ))
        else:
            logger.info('Module profiling was enabled, but no profiling data was collected.')

    if pred_debug_count > 0:
        logger.info('Average prediction debug stats per sample:')
        for key, total_val in pred_debug_sums.items():
            logger.info('%s: %.6f' % (key, total_val / pred_debug_count))

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
