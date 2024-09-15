
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=high/fused' --device=3 --fusion='raw' --log_name='raw_5frame' --n_frame=5
python -u carla_train_eval.py --data_root='./data/processed/2lidar=high_4radar=low/fused' --device=3 --fusion='raw' --log_name='raw_5frame' --n_frame=5