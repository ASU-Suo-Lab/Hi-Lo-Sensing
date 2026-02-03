
# On the Resolution–Performance Tradeoff in LiDAR and 4D Radar Fusion for Roadside Perception

This repo is the official implementation of our paper: On the Resolution–Performance Tradeoff in LiDAR and 4D Radar Fusion for Roadside Perception as well as the follow-ups. Our code is built upon the codebase of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

<p>
  <img src="assets/performance_cost_comparsion.jpeg" width="49%" />
  <img src="assets/overrall comparison.jpg" width="49%" />
</p>


## Overview
- [🤔 Introduction](#introduction)
- [🛠️ Quick Start](#quick-start)


## Introduction
Through systematic model architecture exploration, this paper demonstrates that fusing information-rich 4D mmWave radar with low-resolution LiDAR point clouds can enrich learned feature representations and outperform a high-resolution LiDAR-only baseline in detecting diverse traffic participants. Through extensive ablation studies and experimental evaluation, we demonstrate that the optimal proposed fusion model integrating 4D radar with low-resolution LiDAR achieves object detection accuracy close to a mid-resolution LiDAR-only solution. Furthermore, the fusion of 4D radar and mid-resolution LiDAR even outperforms the solution using high-resolution LiDAR only by `+0.8%`. 
<div align="center">
  <img src="assets/general fusion framework.jpg" width="100%"/>
</div>

We construct a multi-modal, multi-resolution roadside perception dataset in the [CARLA](https://github.com/carla-simulator/carla) simulation environment. To the best of our knowledge, this dataset is the first large-scale roadside perception benchmark dataset combining multi-resolution and multimodal sensors, enabling fair and controlled comparisons of various multimodal fusion strategies.
<div align="center">
  <img src="assets/digital map.jpg" width="100%"/>
</div>

What's more, to improve the fidelity of the simulation platform in capturing the distribution of point-cloud data from traffic participants, particularly for mmWave radar sensing, we develop a lightweight multimodal roadside sensing platform and deploy it on a private roadway to collect real-world multimodal data. These realistic measurements are used to fine-tune the radar module within the simulation environment, while maintaining sensor configurations identical to those used on the physical platform.
<div align="center">
  <img src="assets/Lidar_radar_work_principle.jpg" width="100%"/>
</div>

## Quick Start
### Installation
Please follow [OpenPCDet's official instructions](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) to set up the environment first.

### Dataset Preparation
* Please download our multimodal, multi-resolution dataset, which is divided into three subsets based on resolution: [low-resolution dataset](https://www.dropbox.com/scl/fi/71p8cnrgohstvcxk8g7o9/low_lidar_2radar.zip?rlkey=j9jc8pdln9mzie0mvaawdc52i&st=3odmdk4p&dl=0), [medium-resolution dataset](https://www.dropbox.com/scl/fi/jddck41d8ukkzng2f6r3y/mid_lidar_2radar.zip?rlkey=38k5p5pqrhkvssfdktobd1xro&st=c1oq00au&dl=0), and [high-resolution dataset](https://www.dropbox.com/scl/fi/2gxxdz31tfajhcp6a0nky/high_lidar_2radar.zip?rlkey=svy4zmg2onjir4g4gojyu87qm&st=wmigap1a&dl=0).

* After extracting the files, please organize the downloaded files as follows:
```
OpenPCDet
├── data
│   ├── low_lidar_2radar
│   │   │── v1.0-trainval
│   │   │   │── lidar_fusion
│   │   │   │── radar_fusion
│   │   │   │── label
│   │   │   │── image
│   │   │   │── calib
│   │   │   │── lidar
│   │   │   │── radar
│   │   │   │── trainset.txt
│   │   │   │── valset.txt
│   ├── mid_lidar_2radar
│   │   │── v1.0-trainval
│   │   │   │── ......
│   ├── high_lidar_2radar
│   │   │── v1.0-trainval
│   │   │   │── ......
├── pcdet
├── tools
```

* Run the following command to generate data information. To generate information for the three different resolution datasets, please modify `data_path` to the path of the corresponding dataset and execute the command separately:

```python 
# For example, create train and val info file of low-resolution dataset
python create_carla_dataset.py --data_path /data/low_lidar_2radar/v1.0-trainval --mode train
python create_carla_dataset.py --data_path /data/low_lidar_2radar/v1.0-trainval --mode val
``` 

* The final format of the generated data is as follows:
```
OpenPCDet
├── data
│   ├── low_lidar_2radar
│   │   │── v1.0-trainval
│   │   │   │── lidar_fusion
│   │   │   │── radar_fusion
│   │   │   │── label
│   │   │   │── image
│   │   │   │── calib
│   │   │   │── lidar
│   │   │   │── radar
│   │   │   │── trainset.txt
│   │   │   │── valset.txt
│   │   │   │── carla_gt_database
│   │   │   │── carla_dbinfos_train.pkl
│   │   │   │── carla_infos_train.pkl
│   ├── mid_lidar_2radar
│   │   │── v1.0-trainval
│   │   │   │── ......
│   ├── high_lidar_2radar
│   │   │── v1.0-trainval
│   │   │   │── ......
├── pcdet
├── tools
```

### Training
The configuration files required for training are located in `/tools/cfgs/carla_models`. Before starting training, please ensure that the DATA_PATH field in `/tools/cfgs/dataset_configs/carla_dataset.yaml` is set to the correct dataset directory corresponding to the resolution you intend to train.

To train our final fusion model for low-resolution LiDAR and 4D radar:
```shell
# multi-gpu training
cd tools
sh scripts/dist_train.sh 4 --cfg_file ./cfgs/carla_models/vv_transhead.yaml
```

To train our final fusion model for mid-resolution LiDAR and 4D radar:
```shell
# multi-gpu training
cd tools
sh scripts/dist_train.sh 4 --cfg_file ./cfgs/carla_models/ll_msattn.yaml
```


