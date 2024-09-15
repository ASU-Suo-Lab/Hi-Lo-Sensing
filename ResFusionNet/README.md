
This repository contains the official source code for the research paper [High and Low Resolution Tradeoffs in Roadside Multimodal Sensing]().

For detailed information on script arguments, please refer to the corresponding Python files.

## 1. Point Cloud Preprocessing from CARLA

To preprocess point cloud data from CARLA, execute the following command:

```bash
python pre_process_carla.py
```

## 2. ResFusionNet Training

To train the ResFusionNet model, use the following command:

```bash
python carla_train_eval.py
```

## 3. Model Testing

To test the model using a checkpoint, run:

```bash
python -u carla_test.py --data_root='./your/data/root' --ckpt_path='/path/to/your/checkpoint'
```

## Acknowledgements

This project incorporates code from [this repository](https://github.com/zhulf0804/PointPillars). Please adhere to their installation and compilation guidelines.
