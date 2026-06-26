# DINOv2/NVS

Normal Variation Subspace (NVS) experiments for DINOv2 patch features.

This directory intentionally lives under `DINOv2/` because the current NVS
prototype reuses the local `dinov2_mvtec_nn.py` feature extraction and memory
retrieval utilities.

## Stage-1 experiments

- E1: normal transformations can trigger false positives in DINOv2 nearest
  memory retrieval.
- E2: residuals after removing a normal-variation subspace can separate
  transformed-normal patches from real-defect patches.

Run examples on the server:

```bash
cd ~/yyk/DINOv2

python nvs/e1_transform_fp.py \
  --config nvs/configs/mvtec_dev5_e1.yaml \
  --data-root /home/ubuntu/yyk/datasets/mvtec \
  --device cuda

python nvs/e2_subspace_residual.py \
  --config nvs/configs/mvtec_dev5_e2.yaml \
  --data-root /home/ubuntu/yyk/datasets/mvtec \
  --device cuda
```

