import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

def setup_configurations(args, setup_seed, get_carla_dataset, get_dataloaders, Loss, Carla):
    now = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    now += '_' + args.log_name
    if args.large:
        now += '_large'
    setup_seed(seed=args.seed)
    
    if args.large:
        x_l = 248
        y_l = 248
        voxel_size=[0.32, 0.32, 4]
        anchor_range = [-37, -50, 0.3, 40, 29, 0.3]
        point_cloud_range = [-37, -50, -0.1, 40, 29, 3.9] # Big
    else:    
        # intersection
        x_l = 184
        y_l = 184
        voxel_size=[0.2, 0.2, 4]
        anchor_range = [-20, -26, 0.3, 16, 10, 0.3]
        point_cloud_range = [-20, -26, -0.1, 16, 10, 3.9] # Intersection
    
    

    # Get datasets and dataloaders
    train_dataset, val_dataset, test_dataset = get_carla_dataset(data_root=args.data_root, pcr=point_cloud_range, n_frame=args.n_frame)
    print(f'Num of data samples | train_dataset: {len(train_dataset)} | val_dataset: {len(val_dataset)} | test_dataset: {len(test_dataset)}')
    print(f'Train: Ped: {train_dataset.pedestrian_count} | Cyc: {train_dataset.cyclist_count} | Car: {train_dataset.car_count}')
    print(f'Val: Ped: {val_dataset.pedestrian_count} | Cyc: {val_dataset.cyclist_count} | Car: {val_dataset.car_count}')
    print(f'Test: Ped: {test_dataset.pedestrian_count} | Cyc: {test_dataset.cyclist_count} | Car: {test_dataset.car_count}')
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        dataset={'train': train_dataset, 'val': val_dataset, 'test': test_dataset},
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    from model import CarlaPointPillarsMultiFrame as PointPillars
    
    # Set device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Classes and labels
    CLASSES = Carla.CLASSES
    LABEL2CLASSES = {v: k for k, v in CLASSES.items()}
    
    
    # Initialize model
    pointpillars = PointPillars(args=args, x_l=x_l, y_l=y_l, voxel_size=voxel_size, anchor_range=anchor_range, point_cloud_range=point_cloud_range, nclasses=args.nclasses, n_frame=args.n_frame)
    if not args.no_cuda:
        pointpillars = pointpillars.to(device)
    
    # Loss function
    loss_func = Loss()

    # Optimizer and scheduler
    max_iters = len(train_dataloader) * args.max_epoch
    init_lr = args.init_lr
    optimizer = torch.optim.AdamW(
        params=pointpillars.parameters(), 
        lr=init_lr, 
        betas=(0.95, 0.99),
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,  
        max_lr=init_lr * 10, 
        total_steps=max_iters, 
        pct_start=0.4, 
        anneal_strategy='cos',
        cycle_momentum=True, 
        base_momentum=0.95 * 0.895, 
        max_momentum=0.95,
        div_factor=10
    )
    
    if len(args.log_name) > 0:
        # Create directories for logs and checkpoints
        saved_logs_path = os.path.join(args.saved_path, 'summary', args.data_root.split('/')[-2], now)
        os.makedirs(saved_logs_path, exist_ok=True)
        writer = SummaryWriter(saved_logs_path)

        saved_ckpt_path = os.path.join(args.saved_path, 'checkpoints', args.data_root.split('/')[-2], now)
        os.makedirs(saved_ckpt_path, exist_ok=True)

        saved_print_path = os.path.join(args.saved_path, 'logs', args.data_root.split('/')[-2], now)
        os.makedirs(saved_print_path, exist_ok=True)
        with open(os.path.join(saved_print_path, 'eval_results.txt'), 'w') as f:
            pass
    else:
        writer = None
        saved_logs_path = None
        saved_ckpt_path = None
        saved_print_path = None

    print(f'saved_logs_path: {saved_logs_path}')
    print(f'saved_ckpt_path: {saved_ckpt_path}')
    print(f'saved_print_path: {saved_print_path}')
    
    best_results = {
        "val_0.25": {
            "Pedestrian": [0, 0, 0],
            "Cyclist": [0, 0, 0],
            "Car": [0, 0, 0],
        }, 
        "test_0.25": {
            "Pedestrian": [0, 0, 0],
            "Cyclist": [0, 0, 0],
            "Car": [0, 0, 0],
        },
        "val_0.5": {
            "Pedestrian": [0, 0, 0],
            "Cyclist": [0, 0, 0],
            "Car": [0, 0, 0],
        }, 
        "test_0.5": {
            "Pedestrian": [0, 0, 0],
            "Cyclist": [0, 0, 0],
            "Car": [0, 0, 0],
        },
        "val_0.7": {
            "Pedestrian": [0, 0, 0],
            "Cyclist": [0, 0, 0],
            "Car": [0, 0, 0],
        }, 
        "test_0.7": {
            "Pedestrian": [0, 0, 0],
            "Cyclist": [0, 0, 0],
            "Car": [0, 0, 0],
        },
        "val_mean": {
            "Pedestrian": [0, 0, 0],
            "Cyclist": [0, 0, 0],
            "Car": [0, 0, 0],
        },
        "test_mean": {
            "Pedestrian": [0, 0, 0],
            "Cyclist": [0, 0, 0],
            "Car": [0, 0, 0],
        }
    }
    

    return (best_results, now, train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader, 
            device, CLASSES, LABEL2CLASSES, pointpillars, loss_func, optimizer, scheduler, writer, 
            saved_ckpt_path, saved_print_path)