import os
import yaml
import copy
import time
import pickle
import random
import argparse
import numpy as np
import open3d as o3d
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from os import path as osp
from dataclasses import dataclass
from typing import List, Tuple, Set
from scipy.spatial.transform import Rotation as R


def clear_data_info_unused_keys(data_info):
    keys = list(data_info.keys())
    empty_flag = True
    for key in keys:
        # we allow no annotations in datainfo
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag is False:
                empty_flag = False
            else:
                # sub field is empty
                del data_info[key]
        else:
            empty_flag = False

    return data_info, empty_flag



def clear_instance_unused_keys(instance):
    keys = list(instance.keys())
    for k in keys:
        if instance[k] is None:
            del instance[k]
    return instance


def get_empty_instance():
    """Empty annotation for single instance."""
    instance = dict(
        # (list[float], required): list of 4 numbers representing
        # the bounding box of the instance, in (x1, y1, x2, y2) order.
        bbox=None,
        # (int, required): an integer in the range
        # [0, num_categories-1] representing the category label.
        bbox_label=None,
        #  (list[float], optional): list of 7 (or 9) numbers representing
        #  the 3D bounding box of the instance,
        #  in [x, y, z, w, h, l, yaw]
        #  (or [x, y, z, w, h, l, yaw, vx, vy]) order.
        bbox_3d=None,
        # (bool, optional): Whether to use the
        # 3D bounding box during training.
        bbox_3d_isvalid=None,
        # (int, optional): 3D category label
        # (typically the same as label).
        bbox_label_3d=None,
        # (float, optional): Projected center depth of the
        # 3D bounding box compared to the image plane.
        depth=None,
        #  (list[float], optional): Projected
        #  2D center of the 3D bounding box.
        center_2d=None,
        # (int, optional): Attribute labels
        # (fine-grained labels such as stopping, moving, ignore, crowd).
        attr_label=None,
        # (int, optional): The number of LiDAR
        # points in the 3D bounding box.
        num_lidar_pts=None,
        # (int, optional): The number of Radar
        # points in the 3D bounding box.
        num_radar_pts=None,
        # (int, optional): Difficulty level of
        # detecting the 3D bounding box.
        difficulty=None,
        unaligned_bbox_3d=None)
    return instance


def get_empty_img_info():
    img_info = dict(
        # (str, required): the path to the image file.
        img_path=None,
        # (int) The height of the image.
        height=None,
        # (int) The width of the image.
        width=None,
        # (str, optional): Path of the depth map file
        depth_map=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to image with
        # shape [3, 3], [3, 4] or [4, 4].
        cam2img=None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to image with shape [4, 4].
        lidar2img=None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to ego-vehicle
        # with shape [4, 4].
        cam2ego=None)
    return img_info



def get_single_image_sweep(camera_types):
    single_image_sweep = dict(
        # (float, optional) : Timestamp of the current frame.
        timestamp=None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        ego2global=None)
    # (dict): Information of images captured by multiple cameras
    images = dict()
    for cam_type in camera_types:
        images[cam_type] = get_empty_img_info()
    single_image_sweep['images'] = images
    return single_image_sweep


def get_empty_lidar_points():
    lidar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of LiDAR data file.
        lidar_path=None,
        # (list[list[float]], optional): Transformation matrix
        # from lidar to ego-vehicle
        # with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        lidar2ego=None,
    )
    return lidar_points


def get_empty_radar_points():
    radar_points = dict(
        # (int, optional) : Number of features for each point.
        num_pts_feats=None,
        # (str, optional): Path of RADAR data file.
        radar_path=None,
        # Transformation matrix from lidar to
        # ego-vehicle with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        radar2ego=None,
    )
    return radar_points


def get_empty_standard_data_info(
        camera_types=['CAM0', 'CAM1', 'CAM2', 'CAM3', 'CAM4']):
    data_info = dict(
        # (str): Sample id of the frame.
        sample_idx=None,
        # (str, optional): '000010'
        token=None,
        **get_single_image_sweep(camera_types),
        # (dict, optional): dict contains information
        # of LiDAR point cloud frame.
        lidar_points=get_empty_lidar_points(),
        # (dict, optional) Each dict contains
        # information of Radar point cloud frame.
        radar_points=get_empty_radar_points(),
        # (list[dict], optional): Image sweeps data.
        image_sweeps=[],
        lidar_sweeps=[],
        instances=[],
        # (list[dict], optional): Required by object
        # detection, instance  to be ignored during training.
        instances_ignore=[],
        # (str, optional): Path of semantic labels for each point.
        pts_semantic_mask_path=None,
        # (str, optional): Path of instance labels for each point.
        pts_instance_mask_path=None)
    return data_info



@dataclass
class Pose:
    x: float
    y: float
    z: float
    # degree
    pitch: float
    yaw: float
    roll: float

    @classmethod
    def from_list(cls, args: List[float]) -> "Pose":
        assert len(args) == 6
        return cls(*args)

    def get_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        r = R.from_euler(
            "zyx", (self.yaw, self.pitch, self.roll), degrees=True
        )
        t = np.asarray([self.x, self.y, self.z])

        return r.as_matrix(), t


@dataclass
class LidarInfo:
    index: int
    sensor_rot: np.ndarray
    sensor_trans: np.ndarray
    pc_path: str
    velocity_path: str


@dataclass
class VehicleInfo:
    id: int
    rotation: List[float]  # pitch, yaw, roll
    center: List[float]
    extent: List[float]
    location: List[float]
    srcs: List[int]

    def get_bbox(self) -> List[float]:
        center = np.asarray(self.location) + np.asarray(self.center)
        extent = np.asarray(self.extent) * 2

        bottom_center = center.copy()
        bottom_center[2] -= extent[2] / 2 # 这个修正是必要的，已经通过结果验证过
        # bottom_center[0] = -bottom_center[0]

        return bottom_center.tolist() + extent.tolist() + [
            np.deg2rad(self.rotation[1])
        ]


@dataclass
class PedestrianInfo:
    id: int
    rotation: List[float]  # pitch, yaw, roll
    center: List[float]
    extent: List[float]
    location: List[float]
    srcs: List[int]

    def get_bbox(self) -> List[float]:
        center = np.asarray(self.location) + np.asarray(self.center)
        extent = np.asarray(self.extent) * 2

        bottom_center = center.copy()
        bottom_center[2] -= extent[2] / 2 # 这个修正是必要的，已经通过结果验证过
        # bottom_center[0] = -bottom_center[0]

        return bottom_center.tolist() + extent.tolist() + [
            np.deg2rad(self.rotation[1])
        ]


@dataclass
class CyclistInfo:
    id: int
    rotation: List[float]  # pitch, yaw, roll
    center: List[float]
    extent: List[float]
    location: List[float]
    srcs: List[int]

    def get_bbox(self) -> List[float]:
        center = np.asarray(self.location) + np.asarray(self.center)
        extent = np.asarray(self.extent) * 2

        bottom_center = center.copy()
        bottom_center[2] -= extent[2] / 2 # 这个修正是必要的，已经通过结果验证过
        # 因为raw_data.yaml中对x轴反转,因此这里也反转x轴,同理yaw
        # bottom_center[0] = -bottom_center[0]

        return bottom_center.tolist() + extent.tolist() + [
            np.deg2rad(self.rotation[1])
        ]


@dataclass
class DataInfo:
    scene_id: str
    lidars: List[LidarInfo]
    vehicles: List[VehicleInfo]
    pedestrians: List[PedestrianInfo]
    cyclists: List[CyclistInfo]








def update_carla_infos(pkl_path, out_dir, num_pts_feats):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        # time.sleep(5)
    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist'),
    }
    print(f'Reading from input file: {pkl_path}.')
    # data_list = mmengine.load(pkl_path)
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in tqdm(data_list, total=len(data_list)):
        temp_data_info = get_empty_standard_data_info()

        temp_data_info['sample_idx'] = ori_info_dict['scene_id']

        temp_data_info['lidar_points']['num_pts_feats'] = num_pts_feats
        temp_data_info['lidar_points']['lidar_path'] = ori_info_dict['scene_id'] + "_lidar.bin"
        temp_data_info['radar_points']['num_pts_feats'] = num_pts_feats
        temp_data_info['radar_points']['radar_path'] = ori_info_dict['scene_id'] + "_radar.bin"

        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        if anns is not None:
            num_instances = len(anns['name'])
            # 如果没有gt，则跳过
            if num_instances <= 0:
                continue
            instance_list = []
            for instance_id in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox'] = [0,0,100,100]

                if anns['name'][instance_id] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(
                        anns['name'][instance_id])
                else:
                    ignore_class_name.add(anns['name'][instance_id])
                    empty_instance['bbox_label'] = -1

                loc = anns['bboxes_3d'][instance_id][:3]
                dims = anns['bboxes_3d'][instance_id][3:6]
                rots = anns['bboxes_3d'][instance_id][-1]

                gt_bboxes_3d = loc + dims + [rots]
                empty_instance['bbox_3d'] = gt_bboxes_3d
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['truncated'] = [0]
                empty_instance['occluded'] = [0]
                empty_instance['alpha'] = [-10]
                # 只有测试集里面有，训练集没有
                empty_instance['score'] = []
                empty_instance['difficulty'] = anns['difficulty'][instance_id]
                empty_instance = clear_instance_unused_keys(empty_instance)
                instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'carla'
    metainfo['info_version'] = '1.1'
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)

    with open(out_path, "wb") as f:
        pickle.dump(converted_data_info, f)

def process_scene(scene_id, data_path, sensor_type):
    lidar_infos = []
    vehicle_infos = {}
    pedestrian_infos = {}
    cyclist_infos = {}
    
    for i in range(6):
        pcd_type = sensor_type
        if sensor_type in ['fused', 'ff']:
            anno_path = data_path / f"{scene_id}_lidar{i}.yaml"
            pcd_type = 'lidar'
            if not anno_path.exists():
                anno_path = data_path / f"{scene_id}_radar{i}.yaml"     
                pcd_type = 'radar'   
        else:
            anno_path = data_path / f"{scene_id}_{sensor_type}{i}.yaml"
            
        if not anno_path.exists():
            print(f"Annotation file not found: {anno_path}")
            continue

        with open(anno_path, "r") as f:
            anno = yaml.load(f, yaml.SafeLoader)

        lidar_pose = Pose.from_list(anno["lidar_pose"])
        sensor_rot, sensor_trans = lidar_pose.get_transform()
        pc_path = anno_path.stem + ".pcd"
        if not (data_path / pc_path).exists():
            print(f"Sensor {i}'s point cloud file not found: {pc_path}")
            pc_path = None

        velocity_path = f"{scene_id}_{pcd_type}{i}.bin" if pcd_type == 'radar' else None

        lidar_infos.append(
            LidarInfo(
                index=i,
                sensor_rot=sensor_rot,
                sensor_trans=sensor_trans,
                pc_path=pc_path,
                velocity_path=velocity_path
            )
        )

        for v_id, info in anno["vehicles"].items():
            if v_id not in vehicle_infos:
                vehicle_infos[v_id] = VehicleInfo(
                    id=v_id,
                    rotation=info["rotation"],
                    center=info["center"],
                    extent=info["extent"],
                    location=info["location"],
                    srcs=[],
                )
            vehicle_infos[v_id].srcs.append(i)

        for p_id, info in anno["pedestrians"].items():
            if p_id not in pedestrian_infos:
                pedestrian_infos[p_id] = PedestrianInfo(
                    id=p_id,
                    rotation=info["rotation"],
                    center=info["center"],
                    extent=info["extent"],
                    location=info["location"],
                    srcs=[],
                )
            pedestrian_infos[p_id].srcs.append(i)

        for c_id, info in anno["cyclists"].items():
            if c_id not in cyclist_infos:
                cyclist_infos[c_id] = CyclistInfo(
                    id=c_id,
                    rotation=info["rotation"],
                    center=info["center"],
                    extent=info["extent"],
                    location=info["location"],
                    srcs=[],
                )
            cyclist_infos[c_id].srcs.append(i)

    if len(lidar_infos) == 0:
        print(f"All sensor data not found for {scene_id}")
        return None

    vehicle_infos = list(vehicle_infos.values())
    pedestrian_infos = list(pedestrian_infos.values())
    cyclist_infos = list(cyclist_infos.values())

    return DataInfo(
        scene_id=scene_id,
        lidars=lidar_infos,
        vehicles=vehicle_infos,
        pedestrians=pedestrian_infos,
        cyclists=cyclist_infos
    )


def load_raw_data_infos(data_path: Path, sensor_type: str = 'lidar', num_workers=5) -> List[DataInfo]:
    # scene_ids = set([
    #     anno_path.name[:6]
    #     for anno_path in data_path.glob("*.yaml")
    # ])
    scene_ids = set([
        anno_path.name.split("_")[0]
        for anno_path in data_path.glob("*.yaml")
    ])

    # 定义自定义的排序函数
    def custom_sort(num):
        return num.zfill(7)  # 假设字符串长度为6

    # 将集合转换为列表并排序，使用自定义排序函数
    scene_ids = sorted(list(scene_ids), key=custom_sort)

    # 将排序后的列表中的每个数字转换回字符串
    scene_ids = [str(num) for num in scene_ids]
    
    # scene_ids = scene_ids[:10]
    # print(f'scene_ids: {scene_ids}')

    data_infos = []
    # for scene_id_idx, scene_id in tqdm(enumerate(scene_ids), total=len(scene_ids)): # we use 5 frames for feature-level fusion
        
    #     lidar_infos = []
    #     vehicle_infos = {}
    #     pedestrian_infos = {}
    #     cyclist_infos = {}
    #     valid = True

    #     for i in range(4):
            
    #         pcd_type = sensor_type
    #         if sensor_type in ['fused', 'ff']:
    #             anno_path = data_path / f"{scene_id}_lidar{i}.yaml"
    #             pcd_type = 'lidar'
    #             if not anno_path.exists():
    #                 anno_path = data_path / f"{scene_id}_radar{i}.yaml"     
    #                 pcd_type = 'radar'   
    #         else:
    #             anno_path = data_path / f"{scene_id}_{sensor_type}{i}.yaml"
                
    #         if not anno_path.exists():
    #             print(f"Annotation file not found: {anno_path}")
    #             continue

    #         with open(anno_path, "r") as f:
    #             anno = yaml.load(f, yaml.SafeLoader)

    #         lidar_pose = Pose.from_list(anno["lidar_pose"])
    #         sensor_rot, sensor_trans = lidar_pose.get_transform()
    #         pc_path = anno_path.stem + ".pcd"
    #         if not (data_path / pc_path).exists():
    #             pc_path = None
    #             print(f"Sensor {i}'s point cloud file not found: {pc_path}")

    #         if pcd_type == 'radar':
    #             velocity_path = f"{scene_id}_{pcd_type}{i}.bin"
    #         else:
    #             velocity_path = None
        
    #         lidar_infos.append(
    #             LidarInfo(
    #                 index=i,
    #                 sensor_rot=sensor_rot,
    #                 sensor_trans=sensor_trans,
    #                 pc_path=pc_path,
    #                 velocity_path=velocity_path
    #             )
    #         )

    #         for v_id, info in anno["vehicles"].items():
    #             if v_id not in vehicle_infos:
    #                 vehicle_infos[v_id] = VehicleInfo(
    #                     id=v_id,
    #                     rotation=info["rotation"],
    #                     center=info["center"],
    #                     extent=info["extent"],
    #                     location=info["location"],
    #                     srcs=[],
    #                 )
    #             vehicle_infos[v_id].srcs.append(i)

    #         for p_id, info in anno["pedestrians"].items():
    #             if p_id not in pedestrian_infos:
    #                 pedestrian_infos[p_id] = PedestrianInfo(
    #                     id=p_id,
    #                     rotation=info["rotation"],
    #                     center=info["center"],
    #                     extent=info["extent"],
    #                     location=info["location"],
    #                     srcs=[],
    #                 )
    #             pedestrian_infos[p_id].srcs.append(i)

    #         for c_id, info in anno["cyclists"].items():
    #             if c_id not in cyclist_infos:
    #                 cyclist_infos[c_id] = CyclistInfo(
    #                     id=c_id,
    #                     rotation=info["rotation"],
    #                     center=info["center"],
    #                     extent=info["extent"],
    #                     location=info["location"],
    #                     srcs=[],
    #                 )
    #             cyclist_infos[c_id].srcs.append(i)

    #     if len(lidar_infos) == 0:
    #         print(f"All sensor data not found for {scene_id}")
    #         assert False

    #     vehicle_infos = list(vehicle_infos.values())
    #     pedestrian_infos = list(pedestrian_infos.values())
    #     cyclist_infos = list(cyclist_infos.values())

    #     data_infos.append(
    #         DataInfo(
    #             scene_id=scene_id,
    #             lidars=lidar_infos,
    #             vehicles=vehicle_infos,
    #             pedestrians=pedestrian_infos,
    #             cyclists=cyclist_infos
    #         )
    #     )


    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_scene, scene_id, data_path, sensor_type): scene_id for scene_id in scene_ids}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing scenes"):
            scene_id = futures[future]
            try:
                result = future.result()
                if result:
                    data_infos.append(result)
            except Exception as e:
                print(f"Error processing {scene_id}: {e}")

    return data_infos




class CarlaConverter:
    def __init__(
            self,
            data_path: Path,
            out_path: Path,
            raw_data_infos: List[DataInfo],
            num_workers: int,
            sensor_type: str
    ):
        self.data_path = data_path
        self.out_path = out_path
        self.raw_data_infos = raw_data_infos
        self.num_workers = num_workers
        self.sensor_type = sensor_type
        self.velodyne_path = self.out_path / "velodyne"
        if not self.velodyne_path.exists():
            self.velodyne_path.mkdir(parents=True)

    def convert(self):
        print('Start converting ...')
        # data_infos = mmengine.track_parallel_progress(
        #     self.convert_one,
        #     tasks=self.raw_data_infos,
        #     nproc=self.num_workers,
        #     # nproc=1,
        # )

        # print(f'raw_data_infos: {len(self.raw_data_infos)}')
        # assert False
        
        data_infos = []
        
        # for data_info in tqdm(self.raw_data_infos, total=len(self.raw_data_infos)):
        #     result = self.convert_one(data_info)
        #     data_infos.append(result)
            
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks to the executor
            futures = {executor.submit(self.convert_one, data_info): data_info for data_info in self.raw_data_infos}
            
            # Use tqdm to monitor the progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing data"):
                data_info = futures[future]
                try:
                    # Get the result from the future
                    result = future.result()
                    # Append the result to the data_infos list
                    data_infos.append(result)
                except Exception as e:
                    print(f"Error processing {data_info}: {e}")
                    
        
        print(f'Before removing None, data_infos: {len(data_infos)}')
        data_infos = [data_info for data_info in data_infos if data_info is not None]
        print(f'After removing None, data_infos: {len(data_infos)}')
        data_infos = sorted(data_infos, key=lambda x: int(x['scene_id']))

        # r = random.Random()
        # r.seed(233)
        # r.shuffle(data_infos)
        # random.shuffle(data_infos)

        # num_train_infos = int(len(data_infos) * 0.8)
        # train_data_infos = data_infos[:num_train_infos]
        # val_data_infos = data_infos[num_train_infos:]

        # with open(self.out_path / "carla_infos_train.pkl", "wb") as f:
        #     pickle.dump(train_data_infos, f)

        # with open(self.out_path / "carla_infos_val.pkl", "wb") as f:
        #     pickle.dump(val_data_infos, f)

        with open(self.out_path / "carla_infos.pkl", "wb") as f:
            pickle.dump(data_infos, f)

        print('\nFinished ...')

    def convert_one(self, raw_data_info: DataInfo):
        lidar_points_list, radar_points_list = [], []
        src_indices = []
        
        for i, lidar_info in enumerate(raw_data_info.lidars):
            # print(f'lidar_info {lidar_info.pc_path} - {i} ...')
            if lidar_info.pc_path is None:
                continue

            pc = o3d.io.read_point_cloud(
                str(self.data_path / lidar_info.pc_path)
            )
            # TODO 已经是世界坐标下，不需要在平移旋转点云

            if self.sensor_type == 'lidar' or 'lidar' in lidar_info.pc_path:
                pc.rotate(lidar_info.sensor_rot)
                pc.translate(lidar_info.sensor_trans)
                
            points = np.asarray(pc.points).astype(np.float64)
            
            if 'lidar' in lidar_info.pc_path:
                lidar_points_list.append(np.concatenate([points, np.zeros_like(points)[:, :1]], axis=-1)) # lidar's velocity is 0
            elif 'radar' in lidar_info.pc_path:
                velocity = np.fromfile(self.data_path / lidar_info.pc_path.replace(".pcd", ".bin"), dtype=np.float64)
                # print(f'velocity: {velocity.shape} | points: {points.shape}')
                radar_points_list.append(np.concatenate([points, velocity[:, np.newaxis]], axis=-1))
            else:
                raise ValueError(f"Unknown sensor type: {lidar_info.pc_path}")
            
            src_indices.append(np.ones((points.shape[0],), dtype=np.int64) * lidar_info.index)

        lidar_points = np.concatenate(lidar_points_list, axis=0)
        radar_points = np.concatenate(radar_points_list, axis=0)
        
        if lidar_points.shape[0] != 0 and radar_points.shape[0] != 0:
            # lidar_points[:, 0] = -lidar_points[:, 0]
            lidar_points.tofile(str(self.velodyne_path / raw_data_info.scene_id) + "_lidar.bin")
            
            # radar_points[:, 0] = -radar_points[:, 0]
            radar_points.tofile(str(self.velodyne_path / raw_data_info.scene_id) + "_radar.bin")

            lidar_points_min, lidar_points_max = lidar_points[:, :3].min(0), lidar_points[:, :3].max(0)
            points_range = np.concatenate([lidar_points_min, lidar_points_max], axis=0)

            gt_bboxes_3d = []
            gt_names = []
            difficulty = []
            srcs = []
            
            for vehicle in raw_data_info.vehicles:
                bbox = vehicle.get_bbox()
                assert len(bbox) == 7
                gt_bboxes_3d.append(bbox)
                gt_names.append("Car")
                difficulty.append(0)
                srcs.append(vehicle.srcs)

            for pedestrian in raw_data_info.pedestrians:
                bbox = pedestrian.get_bbox()
                assert len(bbox) == 7
                gt_bboxes_3d.append(bbox)
                gt_names.append("Pedestrian")
                difficulty.append(0)
                srcs.append(pedestrian.srcs)

            for cyclist in raw_data_info.cyclists:
                bbox = cyclist.get_bbox()
                assert len(bbox) == 7
                gt_bboxes_3d.append(bbox)
                gt_names.append("Cyclist")
                difficulty.append(0)
                srcs.append(cyclist.srcs)

            data_info = {
                "scene_id": raw_data_info.scene_id,
                "annos": {
                    "bboxes_3d": gt_bboxes_3d,
                    "name": gt_names,
                    "difficulty": difficulty,
                    "srcs": srcs,
                },
                "points_range": points_range
            }

            return data_info
        else:
            print(f'No points in {raw_data_info.scene_id}.')
            return None


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    root_path = Path(args.data_root)
    out_dir = Path(args.out_dir)
    
    data_infos = load_raw_data_infos(Path(args.data_root), sensor_type=args.sensor_type, num_workers=args.num_workers)
    
    converter = CarlaConverter(
        root_path, out_dir,
        raw_data_infos=data_infos,
        num_workers=args.num_workers,
        sensor_type=args.sensor_type
    )
    converter.convert()
    
    out_dir = args.out_dir
    update_carla_infos(pkl_path=args.out_dir+"carla_infos.pkl", out_dir=args.out_dir, num_pts_feats=args.num_pts_feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--data_root', default='./data/asu_map/3lidar=low_3radar=high/fused/', help='your data root for carla')
    parser.add_argument('--out_dir', default='./data/processed/3lidar=low_3radar=high/fused/', help='your data root for carla')
    parser.add_argument('--sensor_type', default='fused')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--num_pts_feats', type=int, default=4)
    args = parser.parse_args()

    main(args)
