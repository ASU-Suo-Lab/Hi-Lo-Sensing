
# 48 + 30 = 78


# 8min per epoch
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=5 --fusion='raw' --log_name='raw_4frame' --n_frame=4 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=5 --fusion='raw' --log_name='raw_4frame' --n_frame=4 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=high/fused' --device=5 --fusion='raw' --log_name='raw_4frame' --n_frame=4 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=low/fused' --device=5 --fusion='raw' --log_name='raw_4frame' --n_frame=4 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=5 --fusion='raw' --log_name='raw_4frame' --n_frame=4 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=5 --fusion='raw' --log_name='raw_4frame' --n_frame=4 --remove_velocity

# 5min per epoch
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=5 --fusion='feature' --separate_encoder --nn_res --log_name='feature_2frame_sepenc_nnres' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=5 --fusion='feature' --separate_encoder --nn_res --log_name='feature_2frame_sepenc_nnres' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=high/fused' --device=5 --fusion='feature' --separate_encoder --nn_res --log_name='feature_2frame_sepenc_nnres' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=low/fused' --device=5 --fusion='feature' --separate_encoder --nn_res --log_name='feature_2frame_sepenc_nnres' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=5 --fusion='feature' --separate_encoder --nn_res --log_name='feature_2frame_sepenc_nnres' --n_frame=2
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=5 --fusion='feature' --separate_encoder --nn_res --log_name='feature_2frame_sepenc_nnres' --n_frame=2
