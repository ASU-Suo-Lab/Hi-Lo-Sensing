
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
