

# feature fusion 5 frames
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=1 --fusion='feature' --separate_encoder --nn_res --log_name='feature_5frame_sepenc_nnres' --n_frame=5
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=1 --fusion='feature' --separate_encoder --nn_res --log_name='feature_5frame_sepenc_nnres' --n_frame=5