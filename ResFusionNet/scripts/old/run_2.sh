python -u carla_train_eval.py --device=2 --fusion='raw' --log_name='lidar_frame5' --lidar_only --n_frame=5
python -u carla_train_eval.py --device=2 --fusion='feature' --log_name='feature_5' --n_frame=5
python -u carla_train_eval.py --device=2 --fusion='feature' --log_name='feature_5_nnres' --n_frame=5 --res --nn_res