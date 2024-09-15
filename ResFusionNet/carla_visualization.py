import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R


label2name = {
    0: 'car',
    1: 'pedestrian',
    2: 'cyclist',
}


def load_dict_from_npz(filename):
    """
    Load a dictionary from a .npz file.

    Parameters:
    filename (str): The name of the file to load the dictionary from.

    Returns:
    dict: The loaded dictionary.
    """
    with np.load(filename, allow_pickle=True) as data:
        return dict(data.items())


test_results = load_dict_from_npz('./pillar_logs/test/test_results.npz')
print(test_results.keys())
assert False
scene_id = '043541'
# print(type(test_results[scene_id]))
# print(test_results[scene_id].item().keys())
test_results[scene_id] = test_results[scene_id].item()
pred, gt = test_results[scene_id]["pred"], test_results[scene_id]["gt"]
lidar_pts, radar_pts = gt["lidar_pts"], gt["radar_pts"]
full_pts = np.concatenate([lidar_pts, radar_pts], axis=0)
print(f'full_pts: {full_pts.shape}')

# Visualize the pred bbox
pred_bboxes = []
pred_bboxes_labels = []

for pred_bbox, pred_bbox_label in zip(pred['bboxes'], pred['labels']):
    center = pred_bbox[:3]
    extend = pred_bbox[3:6]
    yaw = pred_bbox[6]
    center[2] += extend[2] / 2
    r = R.from_euler("z", yaw, degrees=False)

    bbox = o3d.geometry.OrientedBoundingBox(
        center=center,
        R=r.as_matrix(),
        extent=extend,
    )
    bbox.color = [0, 1, 0]
    pred_bboxes.append(bbox)
    pred_bboxes_labels.append(label2name[pred_bbox_label])

# Visualize the gt bbox
gt_bboxes = []
gt_bboxes_labels = []

for gt_bbox, gt_bbox_label in zip(gt['bboxes'], gt['labels']):
    center = gt_bbox[:3]
    extend = gt_bbox[3:6]
    yaw = gt_bbox[6]
    center[2] += extend[2] / 2
    r = R.from_euler("z", yaw, degrees=False)

    bbox = o3d.geometry.OrientedBoundingBox(
        center=center,
        R=r.as_matrix(),
        extent=extend,
    )
    bbox.color = [1, 0, 0]
    gt_bboxes.append(bbox)
    gt_bboxes_labels.append(label2name[gt_bbox_label])


# Visualize point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(full_pts[:, :3])

# Create an axis-aligned bounding box (AABB)
# bbox_min = np.array([-61.4, -10.6, -0.1])
# bbox_max = np.array([-35.4, 13.3, 3.9])
bbox_min = np.array([-37, -50, -0.1])
bbox_max = np.array([40, 29, 3.9])

aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox_min, max_bound=bbox_max)
aabb.color = [0, 1, 0]

o3d.visualization.draw([pcd] + pred_bboxes + gt_bboxes + [aabb])
