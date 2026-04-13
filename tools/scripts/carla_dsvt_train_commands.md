# CARLA DSVT Training Commands

These commands use the shared base config:

`cfgs/carla_models/dsvt_pillar.yaml`

The config already enables:

- `OPTIMIZATION.USE_AMP: True`
- `OPTIMIZATION.AMP_DTYPE: bf16`

So training only needs to switch `MODEL.BACKBONE_3D.ATTN_BACKEND`.

## Build Extensions

Run this once before training on the Linux server:

```bash
cd ~/OpenPCDet
python setup.py build_ext --inplace
```

## Direct Commands

2-GPU `mha_legacy` baseline:

```bash
cd ~/OpenPCDet/tools
torchrun --nproc_per_node=2 train.py \
  --launcher pytorch \
  --cfg_file cfgs/carla_models/dsvt_pillar.yaml \
  --extra_tag carla_mha_legacy_train \
  --set MODEL.BACKBONE_3D.ATTN_BACKEND mha_legacy
```

2-GPU fastest implementation:

```bash
cd ~/OpenPCDet/tools
torchrun --nproc_per_node=2 train.py \
  --launcher pytorch \
  --cfg_file cfgs/carla_models/dsvt_pillar.yaml \
  --extra_tag carla_flash_attn_fastest_train \
  --set MODEL.BACKBONE_3D.ATTN_BACKEND flash_attn
```

2-GPU `sdpa` reference run:

```bash
cd ~/OpenPCDet/tools
torchrun --nproc_per_node=2 train.py \
  --launcher pytorch \
  --cfg_file cfgs/carla_models/dsvt_pillar.yaml \
  --extra_tag carla_sdpa_train \
  --set MODEL.BACKBONE_3D.ATTN_BACKEND sdpa
```

Single-GPU `mha_legacy` debug run:

```bash
cd ~/OpenPCDet/tools
python train.py \
  --launcher none \
  --cfg_file cfgs/carla_models/dsvt_pillar.yaml \
  --extra_tag carla_mha_legacy_debug \
  --set MODEL.BACKBONE_3D.ATTN_BACKEND mha_legacy
```

Single-GPU fastest debug run:

```bash
cd ~/OpenPCDet/tools
python train.py \
  --launcher none \
  --cfg_file cfgs/carla_models/dsvt_pillar.yaml \
  --extra_tag carla_flash_attn_fastest_debug \
  --set MODEL.BACKBONE_3D.ATTN_BACKEND flash_attn
```

## Wrapper Script

The repo now includes a helper script:

```bash
cd ~/OpenPCDet/tools
bash scripts/carla_dsvt_train.sh 2 mha_legacy
bash scripts/carla_dsvt_train.sh 2 sdpa
bash scripts/carla_dsvt_train.sh 2 flash_attn
bash scripts/carla_dsvt_train.sh 1 mha_legacy
bash scripts/carla_dsvt_train.sh 1 sdpa
bash scripts/carla_dsvt_train.sh 1 flash_attn
```

Optional custom `extra_tag`:

```bash
bash scripts/carla_dsvt_train.sh 2 flash_attn my_custom_tag
```

Additional `train.py` arguments can be appended after `extra_tag`.

## Eval Commands

Single checkpoint, 2-GPU `mha_legacy` evaluation:

```bash
cd ~/OpenPCDet/tools
torchrun --nproc_per_node=2 test.py \
  --launcher pytorch \
  --cfg_file cfgs/carla_models/dsvt_pillar.yaml \
  --ckpt ../output/carla_models/dsvt_pillar/carla_mha_legacy_train/ckpt/checkpoint_epoch_10.pth \
  --extra_tag carla_mha_legacy_train \
  --eval_tag carla_mha_legacy_eval \
  --set MODEL.BACKBONE_3D.ATTN_BACKEND mha_legacy
```

Single checkpoint, 2-GPU fastest implementation evaluation:

```bash
cd ~/OpenPCDet/tools
torchrun --nproc_per_node=2 test.py \
  --launcher pytorch \
  --cfg_file cfgs/carla_models/dsvt_pillar.yaml \
  --ckpt ../output/carla_models/dsvt_pillar/carla_flash_attn_fastest_train/ckpt/checkpoint_epoch_10.pth \
  --extra_tag carla_flash_attn_fastest_train \
  --eval_tag carla_flash_attn_fastest_eval \
  --set MODEL.BACKBONE_3D.ATTN_BACKEND flash_attn
```

The repo also includes an eval wrapper:

```bash
cd ~/OpenPCDet/tools
bash scripts/carla_dsvt_eval.sh 2 mha_legacy \
  ../output/carla_models/dsvt_pillar/carla_mha_legacy_train/ckpt/checkpoint_epoch_10.pth

bash scripts/carla_dsvt_eval.sh 2 flash_attn \
  ../output/carla_models/dsvt_pillar/carla_flash_attn_fastest_train/ckpt/checkpoint_epoch_10.pth
```

Optional custom tags:

```bash
bash scripts/carla_dsvt_eval.sh 2 flash_attn \
  ../output/carla_models/dsvt_pillar/my_train_tag/ckpt/checkpoint_epoch_10.pth \
  my_train_tag my_eval_tag
```

## Low-Threshold Diagnostic Eval

To check whether predictions are being removed purely by score filtering, the repo also includes:

```bash
cd ~/OpenPCDet/tools
bash scripts/carla_dsvt_eval_low_thresh.sh 2 flash_attn \
  ../output/carla_models/dsvt_pillar/carla_flash_attn_fastest_train/ckpt/checkpoint_epoch_10.pth
```

This forces both of these to `0.0` during eval:

- `MODEL.POST_PROCESSING.SCORE_THRESH`
- `MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH`

If `results_nusc.json` is still empty after this run, the issue is upstream of score-threshold filtering.

For direct `sdpa` diagnostics:

```bash
cd ~/OpenPCDet/tools
bash scripts/carla_dsvt_eval_low_thresh.sh 2 sdpa \
  ../output/carla_models/dsvt_pillar/carla_flash_attn_fastest_train/ckpt/checkpoint_epoch_10.pth \
  carla_flash_attn_fastest_train sdpa_low_thresh_eval
```

## Output Directories

Because these commands keep using `cfgs/carla_models/dsvt_pillar.yaml`, outputs stay under:

```text
~/OpenPCDet/output/carla_models/dsvt_pillar/<extra_tag>/
```

Examples:

- `~/OpenPCDet/output/carla_models/dsvt_pillar/carla_mha_legacy_train/`
- `~/OpenPCDet/output/carla_models/dsvt_pillar/carla_flash_attn_fastest_train/`

## Current Recommendation

- Baseline: `MODEL.BACKBONE_3D.ATTN_BACKEND=mha_legacy`
- Reference optimized backend: `MODEL.BACKBONE_3D.ATTN_BACKEND=sdpa`
- Fastest stable implementation: `MODEL.BACKBONE_3D.ATTN_BACKEND=flash_attn`

The fastest stable path assumes:

- third-party `flash-attn` is installed
- `dsvt_set_ops_cuda` has been built
- fused `get_set_single_shift` is available
- fused packed metadata stays disabled
