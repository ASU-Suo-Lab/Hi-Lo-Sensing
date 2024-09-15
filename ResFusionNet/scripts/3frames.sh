

# feature fusion 3 frames nnres
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=3 --fusion='feature' --separate_encoder --nn_res --log_name='feature_3frame_sepenc_nnres' --n_frame=3
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=3 --fusion='feature' --separate_encoder --nn_res --log_name='feature_3frame_sepenc_nnres' --n_frame=3
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=1 --fusion='feature' --separate_encoder --nn_res --log_name='feature_3frame_sepenc_nnres' --n_frame=3
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=1 --fusion='feature' --separate_encoder --nn_res --log_name='feature_3frame_sepenc_nnres' --n_frame=3
python -u carla_train_eval.py --data_root='./data/processed/3lidar=mid_3radar=low/fused' --device=1 --fusion='feature' --separate_encoder --nn_res --log_name='feature_3frame_sepenc_nnres' --n_frame=3


# raw fusion 3 frames
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=2 --fusion='raw' --log_name='raw_3frame' --n_frame=3 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=2 --fusion='raw' --log_name='raw_3frame' --n_frame=3 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=3 --fusion='raw' --log_name='raw_3frame' --n_frame=3 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=3 --fusion='raw' --log_name='raw_3frame' --n_frame=3 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=mid_3radar=low/fused' --device=3 --fusion='raw' --log_name='raw_3frame' --n_frame=3 --remove_velocity


# feature fusion 3 frames res
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=4 --fusion='feature' --separate_encoder --res --log_name='feature_3frame_sepenc' --n_frame=3
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=4 --fusion='feature' --separate_encoder --res --log_name='feature_3frame_sepenc' --n_frame=3
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=5 --fusion='feature' --separate_encoder --res --log_name='feature_3frame_sepenc' --n_frame=3
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=5 --fusion='feature' --separate_encoder --res --log_name='feature_3frame_sepenc' --n_frame=3
python -u carla_train_eval.py --data_root='./data/processed/3lidar=mid_3radar=low/fused' --device=5 --fusion='feature' --separate_encoder --res --log_name='feature_3frame_sepenc' --n_frame=3