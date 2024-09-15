
# feature fusion 5 frames
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=4 --fusion='feature' --separate_encoder --res --log_name='feature_5frame_sepenc' --n_frame=5
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=4 --fusion='feature' --separate_encoder --res --log_name='feature_5frame_sepenc' --n_frame=5