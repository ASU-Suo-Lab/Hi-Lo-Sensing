import copy
import os
import pickle
from pathlib import Path
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import numpy as np
from ..dataset import DatasetTemplate
from PIL import Image


class CarlaDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False

        self.radar_config = self.dataset_cfg.get('RADAR_CONFIG', None)
        if self.radar_config is not None:
            self.use_radar = self.radar_config.get('USE_RADAR', True)
            self.radar_load_dim = self.radar_config.LOAD_DIM
            self.radar_use_dim = self.radar_config.USE_DIM
        else:
            self.use_radar = False

        self.include_carla_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

    def include_carla_data(self, mode):
        self.logger.info('Loading CARLA dataset')
        carla_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                carla_infos.extend(infos)

        self.infos.extend(carla_infos)
        self.logger.info('Total samples for CARLA dataset: %d' % (len(carla_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_lidar_pts(self, index):
        info = self.infos[index]
        lidar_path = info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        return points

    def load_radar_info(self, input_dict, info):
        radar_path = info['radar_path']
        points = np.fromfile(str(radar_path), dtype=np.float32, count=-1).reshape([-1, self.radar_load_dim])
        points = points[:, self.radar_use_dim]
        input_dict["radar_points"] = points
        return input_dict

    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])

        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict

    def load_camera_info(self, input_dict, info):
        CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK"]
        # 保持相机顺序（如无要求，可直接用 values()）
        cams_dict = info.get("cams", {})
        cams = [cams_dict[k] for k in CAM_ORDER if k in cams_dict] or list(cams_dict.values())

        fields = ("image_paths", "lidar2camera", "lidar2image", "camera_intrinsics", "camera2lidar")
        input_dict.update({f: [c[f] for c in cams] for f in fields})

        images = [Image.open(p) for p in input_dict["image_paths"]]
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size if images else None

        input_dict = self.crop_image(input_dict)
        return input_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_pts(index)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

        if self.use_radar:
            input_dict = self.load_radar_info(input_dict, info)

        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)

        data_dict = self.prepare_data(data_dict=input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

        #self.vis_pc(data_dict, self.use_radar)
        #with open(os.path.join('/home/sding32', f"{input_dict['frame_id']}.pkl"), 'wb') as f:
        #    pickle.dump(data_dict, f)
        return data_dict

    def vis_pc(self,data_dict, use_radar):
            lidar_points = data_dict['points']
            lidar_pcd = o3d.geometry.PointCloud()
            lidar_points_intensity = lidar_points[:,3]
            lidar_points_intensity = np.c_[
                np.ones_like(lidar_points_intensity),
                np.ones_like(lidar_points_intensity),
                np.ones_like(lidar_points_intensity),
            ]
            lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points[:,:3])
            lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_points_intensity)
            if use_radar:
                radar_points = data_dict['radar_points']
                radar_pcd = o3d.geometry.PointCloud()
                radar_points_intensity = np.ones(radar_points.shape[0])
                radar_points_intensity = np.c_[
                    2 * np.ones_like(radar_points_intensity),
                    np.zeros_like(radar_points_intensity),
                    np.zeros_like(radar_points_intensity),
                ]
                radar_pcd.points = o3d.utility.Vector3dVector(radar_points[:,:3])
                radar_pcd.colors = o3d.utility.Vector3dVector(radar_points_intensity)
            bboxes = []
            for j in range(len(data_dict['gt_boxes'])):
                box = data_dict['gt_boxes'][j]
                center = box[:3]
                extend = box[3:6]
                yaw = box[6]
                center[2] += extend[2] / 2
                r = R.from_euler("z", yaw, degrees=False)

                bbox = o3d.geometry.OrientedBoundingBox(
                    center=center,
                    R=r.as_matrix(),
                    extent=extend,
                )
                bbox.color = [0, 1, 0]
                bboxes.append(bbox)
            if use_radar:
                o3d.visualization.draw_geometries([lidar_pcd, radar_pcd] + bboxes)
            else:
                o3d.visualization.draw_geometries([lidar_pcd] + bboxes)

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from . import carla_utils
        from . import sunlakes_utils
        nusc_annos = carla_utils.transform_det_annos_to_nusc_annos(det_annos)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of Carla have been saved to {res_path}')

        from nuscenes.eval.detection.config import config_factory
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)


        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        #metrics_summary = carla_utils.evaluate(self.root_path,output_path, res_path, eval_config)
        metrics_summary = sunlakes_utils.evaluate(self.root_path,output_path, res_path, eval_config)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = carla_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict