
python -u carla_train_eval.py --device=1 --fusion='raw' --log_name='lidar' --lidar_only
python -u carla_train_eval.py --device=1 --fusion='raw' --log_name='raw_5frame' --n_frame=5
python -u carla_train_eval.py --device=1 --fusion='raw' --log_name='raw_5frame_wo_velo' --remove_velocity --n_frame=5
python -u carla_train_eval.py --device=2 --fusion='raw' --log_name='lidar_frame5' --lidar_only --n_frame=5
python -u carla_train_eval.py --device=2 --fusion='feature' --log_name='feature_5' --n_frame=5
python -u carla_train_eval.py --device=2 --fusion='feature' --log_name='feature_5_nnres' --n_frame=5 --res --nn_res
python -u carla_train_eval.py --device=4 --fusion='raw' --log_name='raw'
python -u carla_train_eval.py --device=4 --fusion='feature' --log_name='feature_5frame_wo_velo' --n_frame=5 --remove_velocity
python -u carla_train_eval.py --device=4 --fusion='feature' --log_name='feature_5frame_res' --n_frame=5 --res
python -u carla_train_eval.py --device=6 --fusion='raw' --log_name='raw_wo_velo' --remove_velocity
python -u carla_train_eval.py --device=6 --fusion='feature' --log_name='feature_5frame_res_sepenc' --n_frame=5 --res --separate_encoder
python -u carla_train_eval.py --device=6 --fusion='feature' --log_name='feature_5frame_nnres_sepenc' --n_frame=5 --res --nn_res --separate_encoder