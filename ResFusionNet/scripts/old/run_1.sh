
# python -u carla_train_eval.py --device=1 --fusion='raw' --log_name='lidar' --lidar_only
python -u carla_train_eval.py --device=1 --fusion='raw' --log_name='raw_5frame' --n_frame=5
python -u carla_train_eval.py --device=1 --fusion='raw' --log_name='raw_5frame_wo_velo' --remove_velocity --n_frame=5