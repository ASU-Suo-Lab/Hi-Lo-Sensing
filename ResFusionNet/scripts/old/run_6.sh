
# python -u carla_train_eval.py --device=6 --fusion='raw' --log_name='raw_wo_velo' --remove_velocity
python -u carla_train_eval.py --device=6 --fusion='feature' --log_name='feature_5frame_res_sepenc' --n_frame=5 --res --separate_encoder
python -u carla_train_eval.py --device=6 --fusion='feature' --log_name='feature_5frame_nnres_sepenc' --n_frame=5 --res --nn_res --separate_encoder