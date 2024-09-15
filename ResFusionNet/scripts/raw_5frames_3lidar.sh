
# raw fusion 5 frames
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=high/fused' --device=3 --fusion='raw' --log_name='raw_5frame' --n_frame=5 --remove_velocity
python -u carla_train_eval.py --data_root='./data/processed/3lidar=low_3radar=low/fused' --device=3 --fusion='raw' --log_name='raw_5frame' --n_frame=5 --remove_velocity