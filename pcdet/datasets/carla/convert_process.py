import numba
import numpy as np
import os

#TODO DSZ poly dataset
METAINFO = {
        'classes':
        ('car', 'motorcycle', 'pedestrian', 'bus','truck','construction_vehicle'),
        # ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        #  'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
}


def read_label(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    annotation = {}
    annotation['type'] = np.array([line[0] for line in lines])
    annotation['bbox_label_3d'] = np.array([line[1] for line in lines], dtype=np.int32)
    annotation['dimensions'] = np.array([line[2:5] for line in lines], dtype=np.float32)
    annotation['location'] = np.array([line[5:8] for line in lines], dtype=np.float32)
    annotation['rotation_y'] = np.array([line[8] for line in lines], dtype=np.float32)
    annotation['velocity'] = np.array([line[9:11] for line in lines], dtype=np.float32)
    return annotation


def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError


def get_nuscenes_2d_boxes(gt_boxes, names, velos, img_info, img):
    """Get the 2d / mono3d annotation records for a given `sample_data_token`of nuscenes dataset.
    """
    image_w = img.shape[1]
    image_h = img.shape[0]

    world2cam = np.array(img_info['lidar2cam'])
    camera_intrinsic = np.array(img_info['cam2img'])
    repro_recs = []
    gt_boxes_copy = np.copy(gt_boxes)
    gt_boxes_copy[:, 2] -= gt_boxes_copy[:, 5] / 2
    bboxes_corners = bbox3d2corners(gt_boxes_copy)
    for i in range(gt_boxes.shape[0]):
        world_points = bboxes_corners[i]
        world_points = world_points.T
        world_points = np.r_[
            world_points, [np.ones(world_points.shape[1])]]
        # Transform the points from world space to camera space.
        corners_3d = np.dot(world2cam, world_points)

        # Finally we can use our K matrix to do the actual 3D -> 2D.
        points_2d = np.dot(camera_intrinsic, corners_3d[:3, :])
        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])
        # Project 3d box to 2d.
        points_2d = points_2d.T

        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
            (points_2d[:, 2] > 0.0)
        # convert the screen coords (uv) to integers.
        points_2d = points_2d[points_in_canvas_mask]
        if points_2d.shape[0] != 8:
            # 有的顶点被过滤掉了,pillars_2d不完整则不统计
            continue

        min_x, min_y, max_x, max_y = np.min(points_2d[:, 0]), np.min(points_2d[:, 1]), np.max(points_2d[:, 0]), np.max(
            points_2d[:, 1])

        repro_rec = dict()
        repro_rec['bbox_label'] = METAINFO['classes'].index(names[i])
        repro_rec['bbox_label_3d'] = repro_rec['bbox_label']
        repro_rec['bbox'] = [min_x, min_y, max_x, max_y]
        repro_rec['bbox_3d_isvalid'] = True

        loc = np.copy(gt_boxes[i][:3])
        dim = np.copy(gt_boxes[i][3:6])
        rot = np.copy(gt_boxes[i][6:])
        # dir向量用来求转换到cam坐标下的yaw
        dir_3d = np.array((np.cos(rot) + loc[0]).tolist() + (np.sin(rot) + loc[1]).tolist() + [loc[2], 1])
        dir_2d = np.dot(world2cam, dir_3d)
        loc = np.array([loc[0], loc[1], loc[2], 1])
        loc = np.dot(world2cam, loc).tolist()
        yaw = [-np.arctan2(dir_2d[2] - loc[2], dir_2d[0] - loc[0])]

        dim[[0, 1, 2]] = dim[[0, 2, 1]]  # convert wlh to our lhw
        dim = dim.tolist()
        repro_rec['bbox_3d'] = loc[:3] + dim + yaw

        global_v2d = velos[i]
        global_velo3d = np.array([*global_v2d, 0.0])
        cam_velo3d = np.dot(world2cam[:3, :3], global_velo3d)
        velo = cam_velo3d[0::2].tolist()
        repro_rec['velocity'] = velo

        center_2d_with_depth = np.dot(camera_intrinsic, np.array(loc[:3]).reshape(3, ))
        center_2d_with_depth = np.array([
            center_2d_with_depth[0] / center_2d_with_depth[2],
            center_2d_with_depth[1] / center_2d_with_depth[2],
            center_2d_with_depth[2]])

        repro_rec['center_2d'] = center_2d_with_depth[:2].tolist()
        repro_rec['depth'] = center_2d_with_depth[2].tolist()
        # # normalized center2D + depth
        # if samples with depth < 0 will be removed
        if repro_rec['depth'] <= 0:
            continue

        repro_recs.append(repro_rec)
    return repro_recs



def get_points_num_in_bbox(points, dimensions, location, rotation_y, name):
    '''
    points: shape=(N, 4)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    dimensions: shape=(n, 3)
    location: shape=(n, 3)
    rotation_y: shape=(n, )
    name: shape=(n, )
    return: shape=(n, )
    '''
    indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
        points_in_bboxes_v2(
            points=points,
            dimensions=dimensions,
            location=location,
            rotation_y=rotation_y,
            name=name)
    points_num = np.sum(indices, axis=0)
    non_valid_points_num = [-1] * (n_total_bbox - n_valid_bbox)
    points_num = np.concatenate([points_num, non_valid_points_num], axis=0)
    return np.array(points_num, dtype=np.int32)

def points_in_bboxes_v2(points, dimensions, location, rotation_y, name):
    '''
    points: shape=(N, 4)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    dimensions: shape=(n, 3)
    location: shape=(n, 3)
    rotation_y: shape=(n, )
    name: shape=(n, )
    return:
        indices: shape=(N, n_valid_bbox), indices[i, j] denotes whether point i is in bbox j.
        n_total_bbox: int.
        n_valid_bbox: int, not including 'DontCare'
        bboxes_lidar: shape=(n_valid_bbox, 7)
        name: shape=(n_valid_bbox, )
    '''
    n_total_bbox = len(dimensions)
    n_valid_bbox = len([item for item in name if item != 'DontCare'])
    location, dimensions = location[:n_valid_bbox], dimensions[:n_valid_bbox]
    rotation_y, name = rotation_y[:n_valid_bbox], name[:n_valid_bbox]
    bboxes_lidar = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1)
    bboxes_corners = bbox3d2corners(bboxes_lidar)
    group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:,:3], frustum_surfaces) # (N, n), N is points num, n is bboxes number
    return indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name


def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |
    y      | /              |  |     | |
    <------|o               | 7 -----| 4
                            |/   o   |/
                            3 ------ 0
    x: front, y: left, z: top
    '''
    # TODO DSZ 这里yaw要取反
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], -bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]],
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]],
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners

def group_rectangle_vertexs(bboxes_corners):
    '''
    bboxes_corners: shape=(n, 8, 3)
    return: shape=(n, 6, 4, 3)
    '''
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]], axis=1) # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]], axis=1) # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]], axis=1) # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]], axis=1) # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]], axis=1) # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs

def group_plane_equation(bbox_group_rectangle_vertexs):
    '''
    bbox_group_rectangle_vertexs: shape=(n, 6, 4, 3)
    return: shape=(n, 6, 4)
    '''
    # 1. generate vectors for a x b
    vectors = bbox_group_rectangle_vertexs[:, :, :2] - bbox_group_rectangle_vertexs[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1]) # (n, 6, 3)
    normal_d = np.einsum('ijk,ijk->ij', bbox_group_rectangle_vertexs[:, :, 0], normal_vectors) # (n, 6)
    plane_equation_params = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    return plane_equation_params


@numba.jit(nopython=True)
def points_in_bboxes(points, plane_equation_params):
    '''
    points: shape=(N, 3)
    plane_equation_params: shape=(n, 6, 4)
    return: shape=(N, n), bool
    '''
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks