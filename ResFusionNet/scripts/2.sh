
# 30 + 24 = 54

# 4min per epoch
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=2 --fusion='raw' --log_name='raw_2frame' --n_frame=2 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=2 --fusion='raw' --log_name='raw_2frame' --n_frame=2 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=high/fused' --device=2 --fusion='raw' --log_name='raw_2frame' --n_frame=2 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=low/fused' --device=2 --fusion='raw' --log_name='raw_2frame' --n_frame=2 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=2 --fusion='raw' --log_name='raw_2frame' --n_frame=2 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=2 --fusion='raw' --log_name='raw_2frame' --n_frame=2 --remove_velocity


# 5min per epoch
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=2 --fusion='feature' --separate_encoder --res --log_name='feature_2frame_sepenc' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=2 --fusion='feature' --separate_encoder --res --log_name='feature_2frame_sepenc' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=high/fused' --device=2 --fusion='feature' --separate_encoder --res --log_name='feature_2frame_sepenc' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=low/fused' --device=2 --fusion='feature' --separate_encoder --res --log_name='feature_2frame_sepenc' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=2 --fusion='feature' --separate_encoder --res --log_name='feature_2frame_sepenc' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=2 --fusion='feature' --separate_encoder --res --log_name='feature_2frame_sepenc' --n_frame=2

