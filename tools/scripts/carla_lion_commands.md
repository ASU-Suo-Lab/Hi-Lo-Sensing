# CARLA LION Commands

These commands use the shared base config:

`cfgs/carla_models/lion_lidar.yaml`

The config enables:

- `OPTIMIZATION.USE_AMP: True`
- `OPTIMIZATION.AMP_DTYPE: bf16`

The two main comparison modes are:

- `Mamba`: stable baseline
- `FLA_GLA`: fastest stable implementation, with fused mapping enabled

## Build Extensions

Run this once before profiling or training on the Linux server:

```bash
cd ~/OpenPCDet
python setup.py build_ext --inplace
```

## Stable Profile Commands

Stable `Mamba` profile:

```bash
cd ~/OpenPCDet/tools
python profile_lion.py \
  --cfg_file cfgs/carla_models/lion_lidar.yaml \
  --warmup 6 \
  --steps 6 \
  --save_dir ../output/lion_profile/lion_mamba_stable \
  --set MODEL.BACKBONE_3D.OPERATOR.NAME Mamba MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING False
```

Stable `FLA_GLA` profile:

```bash
cd ~/OpenPCDet/tools
python profile_lion.py \
  --cfg_file cfgs/carla_models/lion_lidar.yaml \
  --warmup 6 \
  --steps 6 \
  --save_dir ../output/lion_profile/lion_fla_stable \
  --set MODEL.BACKBONE_3D.OPERATOR.NAME FLA_GLA MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING True
```

Wrapper script:

```bash
cd ~/OpenPCDet/tools
bash scripts/carla_lion_profile.sh Mamba
bash scripts/carla_lion_profile.sh FLA_GLA
```

Optional custom save dir:

```bash
bash scripts/carla_lion_profile.sh FLA_GLA ../output/lion_profile/my_fla_run
```

## Stable Training Commands

2-GPU `Mamba` baseline training:

```bash
cd ~/OpenPCDet/tools
torchrun --nproc_per_node=2 train.py \
  --launcher pytorch \
  --cfg_file cfgs/carla_models/lion_lidar.yaml \
  --extra_tag carla_lion_mamba_train \
  --set MODEL.BACKBONE_3D.OPERATOR.NAME Mamba MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING False
```

2-GPU `FLA_GLA` fastest training:

```bash
cd ~/OpenPCDet/tools
torchrun --nproc_per_node=2 train.py \
  --launcher pytorch \
  --cfg_file cfgs/carla_models/lion_lidar.yaml \
  --extra_tag carla_lion_fla_train \
  --set MODEL.BACKBONE_3D.OPERATOR.NAME FLA_GLA MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING True
```

Single-GPU debug runs:

```bash
cd ~/OpenPCDet/tools
python train.py \
  --launcher none \
  --cfg_file cfgs/carla_models/lion_lidar.yaml \
  --extra_tag carla_lion_mamba_debug \
  --set MODEL.BACKBONE_3D.OPERATOR.NAME Mamba MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING False
```

```bash
cd ~/OpenPCDet/tools
python train.py \
  --launcher none \
  --cfg_file cfgs/carla_models/lion_lidar.yaml \
  --extra_tag carla_lion_fla_debug \
  --set MODEL.BACKBONE_3D.OPERATOR.NAME FLA_GLA MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING True
```

Wrapper script:

```bash
cd ~/OpenPCDet/tools
bash scripts/carla_lion_train.sh 2 Mamba
bash scripts/carla_lion_train.sh 2 FLA_GLA
bash scripts/carla_lion_train.sh 1 Mamba
bash scripts/carla_lion_train.sh 1 FLA_GLA
```

Optional custom `extra_tag`:

```bash
bash scripts/carla_lion_train.sh 2 FLA_GLA my_lion_fla_run
```

## Stable Evaluation Commands

2-GPU `Mamba` evaluation:

```bash
cd ~/OpenPCDet/tools
torchrun --nproc_per_node=2 test.py \
  --launcher pytorch \
  --cfg_file cfgs/carla_models/lion_lidar.yaml \
  --ckpt ../output/carla_models/lion_lidar/carla_lion_mamba_train/ckpt/checkpoint_epoch_10.pth \
  --extra_tag carla_lion_mamba_train \
  --eval_tag carla_lion_mamba_eval \
  --set MODEL.BACKBONE_3D.OPERATOR.NAME Mamba MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING False
```

2-GPU `FLA_GLA` evaluation:

```bash
cd ~/OpenPCDet/tools
torchrun --nproc_per_node=2 test.py \
  --launcher pytorch \
  --cfg_file cfgs/carla_models/lion_lidar.yaml \
  --ckpt ../output/carla_models/lion_lidar/carla_lion_fla_train/ckpt/checkpoint_epoch_10.pth \
  --extra_tag carla_lion_fla_train \
  --eval_tag carla_lion_fla_eval \
  --set MODEL.BACKBONE_3D.OPERATOR.NAME FLA_GLA MODEL.BACKBONE_3D.OPERATOR.USE_FUSED_MAPPING True
```

Wrapper script:

```bash
cd ~/OpenPCDet/tools
bash scripts/carla_lion_eval.sh 2 Mamba \
  ../output/carla_models/lion_lidar/carla_lion_mamba_train/ckpt/checkpoint_epoch_10.pth

bash scripts/carla_lion_eval.sh 2 FLA_GLA \
  ../output/carla_models/lion_lidar/carla_lion_fla_train/ckpt/checkpoint_epoch_10.pth
```

Optional custom `extra_tag` and `eval_tag`:

```bash
bash scripts/carla_lion_eval.sh 2 FLA_GLA \
  ../output/carla_models/lion_lidar/my_lion_fla_run/ckpt/checkpoint_epoch_10.pth \
  my_lion_fla_run my_lion_fla_eval
```

## Stable Result Summary

Current stable comparison:

- `Mamba` baseline trace: [sg046_3485843.1775347746600469897.pt.trace.json](/E:/openpcdet_carla/sg046_3485843.1775347746600469897.pt.trace.json)
- `FLA_GLA + fused mapping` stable trace: [sg046_3484547.1775347359691940238.pt.trace.json](/E:/openpcdet_carla/sg046_3484547.1775347359691940238.pt.trace.json)

Steady-state speedup:

- `Mamba`: `LION/forward = 207.87 ms/step`
- `FLA_GLA + fused mapping`: `LION/forward = 166.21 ms/step`
- Stable gain: about `20%`

## Output Directories

Outputs remain under:

```text
~/OpenPCDet/output/carla_models/lion_lidar/<extra_tag>/
```

Examples:

- `~/OpenPCDet/output/carla_models/lion_lidar/carla_lion_mamba_train/`
- `~/OpenPCDet/output/carla_models/lion_lidar/carla_lion_fla_train/`
