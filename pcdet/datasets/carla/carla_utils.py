"""
The NuScenes data pre-processing and evaluation is modified from
https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
"""

import operator
import os
import json
import pickle
import time
from math import inf
from collections import defaultdict
from .convert_process import *
import numpy as np
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList, DetectionMetricData
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

cls_attr_dist = {
    'barrier': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bicycle': {
        'cycle.with_rider': 2791,
        'cycle.without_rider': 8946,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bus': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 9092,
        'vehicle.parked': 3294,
        'vehicle.stopped': 3881,
    },
    'car': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 114304,
        'vehicle.parked': 330133,
        'vehicle.stopped': 46898,
    },
    'construction_vehicle': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 882,
        'vehicle.parked': 11549,
        'vehicle.stopped': 2102,
    },
    'ignore': {
        'cycle.with_rider': 307,
        'cycle.without_rider': 73,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 165,
        'vehicle.parked': 400,
        'vehicle.stopped': 102,
    },
    'motorcycle': {
        'cycle.with_rider': 4233,
        'cycle.without_rider': 8326,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'pedestrian': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 157444,
        'pedestrian.sitting_lying_down': 13939,
        'pedestrian.standing': 46530,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'traffic_cone': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'trailer': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 3421,
        'vehicle.parked': 19224,
        'vehicle.stopped': 1895,
    },
    'truck': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 21339,
        'vehicle.parked': 55626,
        'vehicle.stopped': 11097,
    },
}




def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def boxes_lidar_to_nusenes(det_info):
    boxes3d = det_info['boxes_lidar']
    scores = det_info['score']
    labels = det_info['pred_labels']

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            #boxes3d[k, [4, 3, 5]],  # wlh
            boxes3d[k,3:6],
            quat, label=labels[k], score=scores[k], velocity=velocity,
        )
        box_list.append(box)
    return box_list

def transform_det_annos_to_nusc_annos(det_annos):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det)

        for k, box in enumerate(box_list):
            name = det['name'][k]
            if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos


def format_nuscene_results(metrics, class_names, version='default'):
    result = '----------------Nuscene %s results-----------------\n' % version
    for name in class_names:
        threshs = ', '.join(list(metrics['label_aps'][name].keys()))
        ap_list = list(metrics['label_aps'][name].values())

        err_name = ', '.join([x.split('_')[0] for x in list(metrics['label_tp_errors'][name].keys())])
        error_list = list(metrics['label_tp_errors'][name].values())

        result += f'***{name} error@{err_name} | AP@{threshs}\n'
        result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
        result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
        result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
        result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[key] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    details.update({
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    })

    return result, details


def accumulate(gt_boxes,
               pred_boxes,
               class_name,
               dist_fcn,
               dist_th):
    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()
    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]
    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]
    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.
    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': []}
    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------
    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None
        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx
        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th
        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))
            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)
            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            # print(f"name: {gt_box_match.detection_name}, gt velo: {gt_box_match.velocity}, pred velo: {pred_box.velocity}")
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))
            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))
            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)
        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)
    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()
    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # --------------------------------------------
    # Accumulate.
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)
    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)
    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp
    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------
    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.
        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))
            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]
    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'])


def cal_metrics(cfg, gt_boxes, pred_boxes):
    start_time = time.time()
    metric_data_list = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
            metric_data_list.set(class_name, dist_th, md)
    # -----------------------------------
    # Step 2: Calculate metrics from the data.
    # -----------------------------------
    TP_METRICS = ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']
    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        # Compute APs.
        for dist_th in cfg.dist_ths:
            metric_data = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(metric_data, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)
        # Compute TP metrics.
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, cfg.dist_th_tp)]
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = np.nan
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = np.nan
            else:
                tp = calc_tp(metric_data, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)
    # Compute evaluation time.
    metrics.add_runtime(time.time() - start_time)
    return metrics, metric_data_list


def calc_ap(md: DetectionMetricData, min_recall: float, min_precision: float) -> float:
    """ Calculated average precision. """
    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1
    prec = np.copy(md.precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)


def calc_tp(md: DetectionMetricData, min_recall: float, metric_name: str) -> float:
    """ Calculates true positive errors. """
    first_ind = round(100 * min_recall) + 1  # +1 to exclude the error at min recall.
    last_ind = md.max_recall_ind  # First instance of confidence = 0 is index of max achieved recall.
    if last_ind < first_ind:
        return 1.0  # Assign 1 here. If this happens for all classes, the score for that TP metric will be 0.
    else:
        return float(np.mean(getattr(md, metric_name)[first_ind: last_ind + 1]))  # +1 to include error at max recall.


def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls):
    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'
    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample
    return all_results, meta

def load_gt_from_pkl(data_root, box_cls) -> EvalBoxes:
    all_annotations = EvalBoxes()
    # Load annotations and filter predictions and annotations.
    data_path = os.path.join(data_root, 'carla_infos_val.pkl')
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    for idx, anno in enumerate(data):
        sample_boxes = []
        for i in range(len(anno['gt_boxes'])):
            sample_boxes.append(
                box_cls(
                    sample_token=anno['token'],
                    translation=anno['gt_boxes'][i][:3],
                    size=anno['gt_boxes'][i][3:6],
                    rotation=[np.cos(anno['gt_boxes'][i][6]/2),0,0,np.sin(anno['gt_boxes'][i][6]/2)],
                    velocity=anno['gt_boxes_velocity'][i],
                    num_pts=int(anno['num_lidar_pts'][i]),
                    detection_name= anno['gt_names'][i],
                    detection_score=-1.0,  # GT samples do not have a score.
                    attribute_name=''
                )
            )
        all_annotations.add_boxes(anno['token'], sample_boxes)
    return all_annotations

def load_gt(data_root, box_cls) -> EvalBoxes:
    all_annotations = EvalBoxes()
    # Load annotations and filter predictions and annotations.
    ids_file = os.path.join(data_root, 'valset.txt')
    with open(ids_file, 'r') as f:
        ids = [id.strip() for id in f.readlines()]
    for idx, sample_id in enumerate(ids):
        sample_boxes = []
        label_path = os.path.join(data_root, 'label', f'{sample_id}.txt')
        lidar_path = os.path.join(data_root, 'lidar_fusion', f'{sample_id}.bin')
        lidar_points = read_points(lidar_path, dim=5)
        annos = read_label(label_path)
        label_3d = annos['bbox_label_3d']
        locs = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        names = annos['type']
        velocitys = annos['velocity']
        annos['num_points_in_gt'] = get_points_num_in_bbox(
            points=lidar_points,
            dimensions=dims,
            location=locs,
            rotation_y=rots,
            name=names)
        locs[:, 2] += dims[:, 2] / 2
        for i in range(label_3d.shape[0]):
            sample_boxes.append(
                box_cls(
                    sample_token=sample_id,
                    translation=locs[i],
                    size=dims[i],
                    rotation=[np.cos(rots[i]/2),0,0,np.sin(rots[i]/2)],
                    velocity=velocitys[i],
                    num_pts=int(annos['num_points_in_gt'][i]),
                    detection_name=names[i],
                    detection_score=-1.0,  # GT samples do not have a score.
                    attribute_name=''
                )
            )
        all_annotations.add_boxes(sample_id, sample_boxes)
    return all_annotations


def evaluate(root_path, output_dir, result_path, cfg):
    pred_boxes, meta = load_prediction(result_path, cfg.max_boxes_per_sample, DetectionBox)
    # TODO DSZ poly dataset
    gt_boxes = load_gt_from_pkl(root_path, DetectionBox)
    # Run evaluation.
    metrics, metric_data_list = cal_metrics(cfg, gt_boxes, pred_boxes)

    # Dump the metric data, meta and metrics to disk.
    print('Saving metrics to: %s' % output_dir)
    metrics_summary = metrics.serialize()
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)


    # Print high-level metrics.
    print('mAP: %.4f' % (metrics_summary['mean_ap']))
    err_name_mapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    for tp_name, tp_val in metrics_summary['tp_errors'].items():
        print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
    print('NDS: %.4f' % (metrics_summary['nd_score']))
    print('Eval time: %.1fs' % metrics_summary['eval_time'])

    # Print per-class metrics.
    print()
    print('Per-class results:')
    print('%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' % ('Object Class', 'AP', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE'))
    class_aps = metrics_summary['mean_dist_aps']
    class_tps = metrics_summary['label_tp_errors']
    for class_name in class_aps.keys():
        print('%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f'
              % (class_name, class_aps[class_name],
                 class_tps[class_name]['trans_err'],
                 class_tps[class_name]['scale_err'],
                 class_tps[class_name]['orient_err'],
                 class_tps[class_name]['vel_err'],
                 class_tps[class_name]['attr_err']))

    return metrics_summary

def _xy_distance(box):
    """水平距离 d = sqrt(x^2 + y^2)。根据你的 DetectionBox 字段做兼容。"""
    SCENE_ORIGIN = (18, 12)
    x, y = box.translation[:2]
    return float(np.hypot(x - SCENE_ORIGIN[0], y - SCENE_ORIGIN[1]))

def _in_bin(box, lo, hi):
    d = _xy_distance(box)
    return (lo <= d) and (d < hi)

class _GTBinView:
    """
    仅暴露某距离桶内的 GT：
      - .all : 该桶内所有 GT 列表
      - [sample_token] : 该样本下（在该桶内）的 GT 列表
    """
    def __init__(self, gt_all, lo, hi):
        self._all = [g for g in gt_all if _in_bin(g, lo, hi)]
        self._by_sample = defaultdict(list)
        for g in self._all:
            self._by_sample[g.sample_token].append(g)

    @property
    def all(self):
        return self._all

    def __getitem__(self, sample_token):
        return self._by_sample.get(sample_token, [])

class _PredBinView:
    """
    仅暴露某距离桶内的预测：
      - .all : 该桶内所有预测列表
    """
    def __init__(self, pred_all, lo, hi):
        self._all = [p for p in pred_all if _in_bin(p, lo, hi)]

    @property
    def all(self):
        return self._all

def cal_metrics_by_distance_bins(cfg, gt_boxes, pred_boxes,
                                 bins=((0, 30), (30, 50), (50, inf))):
    """
    复用 cal_metrics：对每个距离桶分别评估并返回字典。
    返回:
      {
        '0-30m':   {'metrics_summary':..., 'metric_data_list':...},
        '30-50m':  {...},
        '50-inf m':{...}
      }
    """
    out = {}
    # 原始容器的“全量列表”
    gt_all   = gt_boxes.all
    pred_all = pred_boxes.all

    for lo, hi in bins:
        name = f"{int(lo)}-{('inf' if hi==inf else int(hi))}m"

        gt_view   = _GTBinView(gt_all, lo, hi)
        pred_view = _PredBinView(pred_all, lo, hi)

        # 如果该桶里没有 GT，给一个空的默认指标（与 nuScenes 行为保持一致）
        if len([1 for g in gt_view.all]) == 0:
            empty_metrics = DetectionMetrics(cfg).serialize()
            out[name] = {'metrics_summary': empty_metrics, 'metric_data_list': DetectionMetricDataList()}
            continue

        # 直接调用你已有的 cal_metrics
        metrics, metric_data_list = cal_metrics(cfg, gt_view, pred_view)
        ms = metrics.serialize()
        out[name] = {'metrics_summary': metrics.serialize(), 'metric_data_list': metric_data_list}

        print(f"\n📊 Distance bin [{name}] results:")
        class_aps = ms["mean_dist_aps"]  # {class_name: ap_value}
        for cls_name, ap in class_aps.items():
            print(f"  {cls_name:<20} mAP = {ap:.4f}")
        print(f"  Overall mAP = {ms['mean_ap']:.4f}")
    return out
