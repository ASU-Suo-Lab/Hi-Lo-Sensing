python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=1 --fusion='raw' --log_name='lidar' --lidar_only
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=1 --fusion='raw' --log_name='raw'
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=1 --fusion='raw' --log_name='raw_wo_velo' --remove_velocity 

python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=1 --fusion='raw' --log_name='lidar' --lidar_only
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=1 --fusion='raw' --log_name='raw'
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=1 --fusion='raw' --log_name='raw_wo_velo' --remove_velocity 

# python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=0 --fusion='raw' --log_name='lidar' --lidar_only
# python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=0 --fusion='raw' --log_name='raw'
# python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=0 --fusion='raw' --log_name='raw_wo_velo' --remove_velocity 

# python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=0 --fusion='raw' --log_name='lidar' --lidar_only
# python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=0 --fusion='raw' --log_name='raw'
# python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=0 --fusion='raw' --log_name='raw_wo_velo' --remove_velocity 