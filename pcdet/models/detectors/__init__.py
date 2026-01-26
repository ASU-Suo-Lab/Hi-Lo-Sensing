from .detector3d_template import Detector3DTemplate
from .transfusion import TransFusion
from .lidar_radar_earlyF import LidarRadar
from .lidar_radar_1 import LidarRadar1
from .lidar_radar_2 import LidarRadar2
from .pillarnet import PillarNet
from .centerpoint import CenterPoint
from .bevfusion import BevFusion

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'TransFusion': TransFusion,
    'LidarRadar': LidarRadar,
    'LidarRadar1': LidarRadar1,
    'LidarRadar2': LidarRadar2,
    'PillarNet': PillarNet,
    'CenterPoint': CenterPoint,
    'BevFusion': BevFusion,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
