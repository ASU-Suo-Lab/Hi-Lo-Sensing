
# python -u carla_train_eval.py --device=4 --fusion='raw' --log_name='raw'
python -u carla_train_eval.py --device=4 --fusion='feature' --log_name='feature_5frame_wo_velo' --n_frame=5 --remove_velocity
python -u carla_train_eval.py --device=4 --fusion='feature' --log_name='feature_5frame_res' --n_frame=5 --res