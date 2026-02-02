
# On the Resolution–Performance Tradeoff in LiDAR and 4D Radar Fusion for Roadside Perception

This repo is the official implementation of our paper: On the Resolution–Performance Tradeoff in LiDAR and 4D Radar Fusion for Roadside Perception as well as the follow-ups. It's built upon the codebase of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

<p>
  <img src="assets/performance_cost_comparsion.jpeg" width="49%" />
  <img src="assets/overrall comparison.jpg" width="49%" />
</p>


## Overview
- [🤔 Introduction](#introduction)
- [🛠️ Quick Start](#quick-start)


## Introduction
Through systematic model architecture exploration, this paper demonstrates that fusing information-rich 4D mmWave radar with low-resolution LiDAR point clouds can enrich learned feature representations and outperform a high-resolution LiDAR-only baseline in detecting diverse traffic participants. Through extensive ablation studies and experimental evaluation, we demonstrate that the optimal proposed fusion model integrating 4D radar with low-resolution LiDAR achieves object detection accuracy close to a mid-resolution LiDAR-only solution. Furthermore, the fusion of 4D radar and mid-resolution LiDAR even outperforms the solution using high-resolution LiDAR only by 0.8\%. 
<div align="center">
  <img src="assets/general fusion framework.jpg" width="100%"/>
</div>

We construct a multi-modal, multi-resolution roadside perception dataset in the [CARLA](https://github.com/carla-simulator/carla) simulation environment. To the best of our knowledge, this dataset is the first large-scale roadside perception benchmark dataset combining multi-resolution and multimodal sensors, enabling fair and controlled comparisons of various multimodal fusion strategies.
<div align="center">
  <img src="assets/digital map.jpg" width="100%"/>
</div>

## Quick Start
### Installation
Please follow [OpenPCDet's official instructions](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) to set up the environment first.

### Dataset Preparation
* Please download our multimodal, multi-resolution dataset, which is divided into three subsets based on resolution: [low-resolution dataset](https://www.dropbox.com/scl/fi/mo084bo79y2a8non8el7p/2025-10-01_22-32-44-low-2radar.zip?rlkey=n9301maba47wwgggifnmmkfb8&st=jg1n77it&dl=0), [medium-resolution dataset](https://www.dropbox.com/scl/fi/ubol0sdci07ehxnd1s5es/2025-10-02_09-12-13-mid-2radar.zip?rlkey=wwmhczc16v07f1gwqkx4cwgce&st=nnavvssp&dl=0), and [high-resolution dataset](https://www.dropbox.com/scl/fi/5150dt742otnotzl5z08k/2025-10-02_13-09-29-high-2radar.zip?rlkey=xgydc7ibp8k6ulg6k3mjj717g&st=vjm6aldz&dl=0).

* After the preprocessing is complete, please organize the downloaded files as follows:
```
OpenPCDet
├── data
│   ├── low_lidar_2radar
│   │   │── v1.0-trainval
│   │   │   │── lidar_fusion
│   │   │   │── radar_fusion
│   │   │   │── label
│   │   │   │── ......
│   ├── mid_lidar_2radar
│   │   │── v1.0-trainval
│   │   │   │── lidar_fusion
│   │   │   │── radar_fusion
│   │   │   │── label
│   │   │   │── ......
│   ├── high_lidar_2radar
│   │   │── v1.0-trainval
│   │   │   │── lidar_fusion
│   │   │   │── radar_fusion
│   │   │   │── label
│   │   │   │── ......
├── pcdet
├── tools
```

* Generate the data infos by running the following command:

```python 
# Create dataset info file, lidar and image gt database
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval \
    --with_cam \
    --with_cam_gt \
``` 

* The final format of the generated data is as follows:
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
│   │   │   │── img_gt_database_10sweeps_withvelo
│   │   │   │── gt_database_10sweeps_withvelo
│   │   │   │── nuscenes_10sweeps_withvelo_lidar.npy (optional) # if open share mem
│   │   │   │── nuscenes_10sweeps_withvelo_img.npy (optional) # if open share mem
│   │   │   │── nuscenes_infos_10sweeps_train.pkl  
│   │   │   │── nuscenes_infos_10sweeps_val.pkl
│   │   │   │── nuscenes_dbinfos_10sweeps_withvelo.pkl
├── pcdet
├── tools
```
