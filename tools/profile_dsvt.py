import _init_path
import argparse
import datetime
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='profile DSVT hotspots')
    parser.add_argument('--cfg_file', type=str, required=True, help='config for profiling')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size per gpu')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--warmup', type=int, default=2, help='warmup steps')
    parser.add_argument('--steps', type=int, default=6, help='profiled steps')
    parser.add_argument('--amp_dtype', choices=['fp16', 'bf16'], default=None, help='autocast dtype')
    parser.add_argument('--save_dir', type=str, default='output/dsvt_profile', help='trace output directory')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='extra config keys')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    args.amp_dtype = args.amp_dtype or cfg.OPTIMIZATION.get('AMP_DTYPE', 'fp16')
    return args, cfg


def main():
    args, cfg = parse_config()
    assert torch.cuda.is_available(), 'CUDA is required for profiling'

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / ('profile_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)

    _, train_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        training=True,
        logger=logger
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_loader.dataset)
    model.cuda().train()
    model_func = model_fn_decorator()

    autocast_dtype = torch.float16 if args.amp_dtype == 'fp16' else torch.bfloat16
    use_amp = cfg.OPTIMIZATION.get('USE_AMP', False)
    use_grad_scaler = use_amp and args.amp_dtype == 'fp16'
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=args.warmup, active=args.steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(save_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    )

    dataloader_iter = iter(train_loader)
    total_steps = args.warmup + args.steps
    profiler.start()
    for _ in range(total_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)

        model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
            loss, _, _ = model_func(model, batch)

        if use_grad_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        profiler.step()

    profiler.stop()
    logger.info(f'Profiler traces written to: {save_dir.resolve()}')
    print(f'Profiler traces written to: {save_dir.resolve()}')


if __name__ == '__main__':
    main()
