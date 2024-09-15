# 60 + 18 = 78


# 3min per epoch
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=3 --fusion='feature' --separate_encoder --res --log_name='feature_1frame_sepenc' --n_frame=1
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=3 --fusion='feature' --separate_encoder --res --log_name='feature_1frame_sepenc' --n_frame=1
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=high/fused' --device=3 --fusion='feature' --separate_encoder --res --log_name='feature_1frame_sepenc' --n_frame=1
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=low/fused' --device=3 --fusion='feature' --separate_encoder --res --log_name='feature_1frame_sepenc' --n_frame=1
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=3 --fusion='feature' --separate_encoder --res --log_name='feature_1frame_sepenc' --n_frame=1
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=3 --fusion='feature' --separate_encoder --res --log_name='feature_1frame_sepenc' --n_frame=1



# 10min per epoch
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=3 --fusion='feature' --separate_encoder --nn_res --log_name='feature_4frame_sepenc_nnres' --n_frame=4
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=3 --fusion='feature' --separate_encoder --nn_res --log_name='feature_4frame_sepenc_nnres' --n_frame=4
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=high/fused' --device=3 --fusion='feature' --separate_encoder --nn_res --log_name='feature_4frame_sepenc_nnres' --n_frame=4
python -u carla_train_eval.py --data_root='./data/processed/2lidar=mid_4radar=low/fused' --device=3 --fusion='feature' --separate_encoder --nn_res --log_name='feature_4frame_sepenc_nnres' --n_frame=4
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=3 --fusion='feature' --separate_encoder --nn_res --log_name='feature_4frame_sepenc_nnres' --n_frame=4
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=3 --fusion='feature' --separate_encoder --nn_res --log_name='feature_4frame_sepenc_nnres' --n_frame=4