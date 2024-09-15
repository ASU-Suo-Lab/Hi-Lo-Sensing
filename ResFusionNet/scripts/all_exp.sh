# radar
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=0 --fusion='raw' --log_name='radar' --radar_only
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=0 --fusion='raw' --log_name='radar' --radar_only
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=0 --fusion='raw' --log_name='radar' --radar_only
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=0 --fusion='raw' --log_name='radar' --radar_only


# radar without velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=0 --fusion='raw' --log_name='radar_wovelo' --radar_only --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=0 --fusion='raw' --log_name='radar_wovelo' --radar_only --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=0 --fusion='raw' --log_name='radar_wovelo' --radar_only --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=0 --fusion='raw' --log_name='radar_wovelo' --radar_only --remove_velocity



# lidar
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=0 --fusion='raw' --log_name='lidar' --lidar_only
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=0 --fusion='raw' --log_name='lidar' --lidar_only
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=0 --fusion='raw' --log_name='lidar' --lidar_only
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=0 --fusion='raw' --log_name='lidar' --lidar_only



# raw fusion (no velocity)
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=0 --fusion='raw' --log_name='raw' --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=0 --fusion='raw' --log_name='raw' --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=0 --fusion='raw' --log_name='raw' --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=0 --fusion='raw' --log_name='raw' --remove_velocity




# raw fusion 5 frames
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=0 --fusion='raw' --log_name='raw_5frame' --n_frame=5 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=0 --fusion='raw' --log_name='raw_5frame' --n_frame=5 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=0 --fusion='raw' --log_name='raw_5frame' --n_frame=5 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=0 --fusion='raw' --log_name='raw_5frame' --n_frame=5 --remove_velocity



# feature fusion 5 frames
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=0 --fusion='feature' --log_name='feature_5frame_sepenc' --n_frame=5
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=0 --fusion='feature' --log_name='feature_5frame_sepenc' --n_frame=5
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=0 --fusion='feature' --log_name='feature_5frame_sepenc' --n_frame=5
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=0 --fusion='feature' --log_name='feature_5frame_sepenc' --n_frame=5



