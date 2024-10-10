# High and Low Resolution Tradeoffs in Roadside Multimodal Sensing
## Content
<!-- - [News](#news-fire) -->
- [Introduction](#Introduction)
- [Multimodal Sensor Deployment under Uncertainties](#Multimodal-Sensor-Deployment-under-Uncertainties)
- [Dataset Download](#Dataset-download)
- [ResFusionNet for 3D Object Detection](#ResFusionNet-for-3D-Object-Detection)
<!-- - [Citation](#citation) -->

## Introduction
High and Low Resolution Tradeoffs in Roadside Multimodal Sensing: This paper underscores the potential of using low spatial resolution but information-rich sensors to enhance detection capabilities for vulnerable road users while highlighting the necessity of thoroughly evaluating sensor modality heterogeneity, traffic participant diversity, and operational uncertainties when making sensor tradeoffs in practical applications.

<p align="center">
<img src="resource/ICRA2025.gif" width="500" alt="" class="img-responsive">
</p>

## Multimodal Sensor Deployment under Uncertainties
This paper introduces a sensor placement algorithm to manage uncertainties in sensor visibility influenced by environmental or human-related factors.

(code will be release soon)

## Dataset Download

Please click on this [link](https://drive.google.com/drive/folders/1Hr_VLnZNa5CrmdpOtwqxZSNFjEr8_N7x?usp=sharing) to download the data.

## ResFusionNet for 3D Object Detection

This paper proposes Residual Fusion Net (ResFusionNet) to fuse multimodal data for 3D object detection, which enables a quantifiable tradeoffbetween spatial resolution and information richness across different modalities.

<details>
<summary>ResFusionNet Details</summary>

### 1. Point Cloud Preprocessing from CARLA

To preprocess point cloud data from CARLA, execute the following command:

```bash
python pre_process_carla.py
```

### 2. ResFusionNet Training

To train the ResFusionNet model, use the following command:

```bash
python carla_train_eval.py
```

### 3. Model Testing

To test the model using a checkpoint, run:

```bash
python -u carla_test.py --data_root='./your/data/root' --ckpt_path='/path/to/your/checkpoint'
```

### Acknowledgements

This project incorporates code from [this repository](https://github.com/zhulf0804/PointPillars). Please adhere to their installation and compilation guidelines.
