# Training & Rendering

## Purpose

Entry points for training avatars from video and evaluating/animating them. `train.py` runs the optimization loop; `render.py` handles test evaluation and out-of-distribution pose animation.

## Key Abstractions

- `training(config)` in `train.py` — Main training loop
- `validation()` in `train.py` — Periodic evaluation on test/train subsets
- `test(config)` in `render.py` — Evaluate trained model on held-out views
- `predict(config)` in `render.py` — Animate with out-of-distribution poses
- `C(iteration, value)` in `train.py` — Iteration-dependent hyperparameter scheduler (step function from list)

## Data Flow

### Training
1. Hydra loads composed config from `configs/`
2. `Scene` created with `GaussianModel` + datasets
3. Each iteration:
   - Random frame sampled (without-replacement epoch)
   - `render()` called → internally runs `scene.convert_gaussians()` (pose correction → non-rigid → rigid → texture) then rasterizes
   - Loss computed: L1 + DSSIM + perceptual (LPIPS) + mask + skinning + AIAP regularization
   - Densification (clone/split/prune) between `densify_from_iter` and `densify_until_iter`
   - Optimizer step for both Gaussians and converter networks

### Evaluation / Prediction
1. Load checkpoint (default: last iteration)
2. Iterate over test dataset, render each frame
3. For test: compute PSNR/SSIM/LPIPS metrics
4. For predict: save rendered images, measure timing

## Loss Terms

| Loss | Config Key | Purpose |
|------|-----------|---------|
| L1 | `lambda_l1` | Pixel reconstruction |
| DSSIM | `lambda_dssim` | Structural similarity |
| Perceptual | `lambda_perceptual` | LPIPS VGG feature matching (foreground-cropped) |
| Mask | `lambda_mask` | Silhouette supervision (BCE or L1) |
| Skinning | `lambda_skinning` | Regularize learned skinning weights toward SMPL |
| AIAP xyz | `lambda_aiap_xyz` | As-rigid-as-possible position preservation |
| AIAP cov | `lambda_aiap_cov` | As-rigid-as-possible covariance preservation |

## Key Files

| File | Role |
|------|------|
| `train.py` | Training loop, loss computation, densification, wandb logging |
| `render.py` | Test evaluation and out-of-distribution pose prediction |
| `configs/config.yaml` | Root config with all defaults and hyperparameters |
| `configs/option/*.yaml` | Iteration count and option overrides |

## Non-obvious Patterns

- **Without-replacement sampling**: `data_stack` pops random indices; refills when empty. This ensures every frame is seen once per epoch. `[OBSERVED]`
- **`C()` scheduler**: Hyperparameters like `lambda_skinning: [10,1000,0.1]` mean "value 10 until iter 1000, then 0.1". Parsed as alternating `[threshold, value]` pairs. `[OBSERVED]`
- **Foreground-cropped perceptual loss**: LPIPS is computed only on the tight bounding box of the foreground mask, not the full image. `[OBSERVED]`
