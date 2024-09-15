python -u carla_train_eval.py --device=5 --fusion='raw' --log_name='lidar' --lidar_only
python -u carla_train_eval.py --device=5 --fusion='raw' --log_name='raw'
python -u carla_train_eval.py --device=5 --fusion='raw' --log_name='raw_wo_velo' --remove_velocity