# CLAUDE.md — 3DGS-Avatar

## Project Overview

3DGS-Avatar: Animatable human avatars using deformable 3D Gaussian Splatting (CVPR 2024). Given monocular video + SMPL body fits, learns 3D Gaussians in canonical pose space and deforms them per-frame.

**Tech stack:** Python 3.7, PyTorch 1.12, CUDA 11.6, Hydra config, SMPL body model.

## Entry Points

- `train.py` — Training loop: loads config via Hydra, creates Scene, iterates over frames with rendering + loss + densification.
- `render.py` — Evaluation and out-of-distribution pose rendering.
- `extract_smpl_parameters.py` — Extracts SMPL parameters from datasets.

## Key Directories

| Directory | Purpose |
|---|---|
| `models/` | Core model components: deformer (rigid/non-rigid), pose correction, texture, gaussian converter |
| `scene/` | Scene orchestration, GaussianModel (learnable point cloud), camera definitions |
| `gaussian_renderer/` | Differentiable Gaussian rasterization interface |
| `dataset/` | Dataset loaders: ZJU-MoCap (`zjumocap.py`, `refined_zjumocap.py`), PeopleSnapshot (`people_snapshot.py`) |
| `utils/` | Loss functions, camera utils, graphics utils, SH utils, general helpers |
| `configs/` | Hydra config hierarchy (see below) |
| `submodules/` | External deps: `diff-gaussian-rasterization`, `simple-knn` |

## Documentation

All architecture docs live in `docs/architecture/`:

- **`overview.md`** — High-level system architecture, subsystem map, data flow, tech stack
- **`gaussian-primitives.md`** — Learnable 3D Gaussian representation, densification strategy (clone/split/prune), color modes
- **`scene-management.md`** — Scene class as coordinator, GaussianConverter pipeline (pose correction → augmentation → non-rigid → rigid → texture)
- **`training-rendering.md`** — Training loop details, all 7 loss terms (L1, DSSIM, perceptual, mask, skinning, AIAP), evaluation/prediction modes

`docs/decisions/` — Reserved for architecture decision records.

## Config System (Hydra)

Root config: `configs/config.yaml`. Uses Hydra composition with these config groups:

| Group | Options | Purpose |
|---|---|---|
| `dataset/` | `zjumocap_377_mono`, `ps_female_3`, etc. | Dataset-specific settings |
| `non_rigid/` | `hashgrid`, `mlp`, `hannw_mlp`, `identity` | Non-rigid deformer architecture |
| `rigid/` | `smpl_nn`, `skinning_field`, `identity` | Rigid deformation method |
| `texture/` | `sh`, `mlp`, `shallow_mlp` | Color/texture network |
| `pose_correction/` | `direct`, `none` | SMPL pose correction |
| `option/` | `iter15k`–`iter50k`, `no_mask`, `no_val`, `test_all` | Training overrides |

Override configs from CLI: `python train.py dataset=zjumocap_386_mono non_rigid=hashgrid option=iter30k`

## Model Pipeline (GaussianConverter)

Canonical Gaussians → Pose Correction → Pose Augmentation → Non-Rigid Deformation → Rigid Deformation (LBS) → Texture → Rendered Image

## Setup

See `README.md` for full install instructions. Key steps:
1. Create conda env from `environment.yml`
2. Install submodules (`diff-gaussian-rasterization`, `simple-knn`)
3. Download SMPL models to `assets/`
4. Prepare dataset (ZJU-MoCap or PeopleSnapshot)

## Common Commands

```bash
# Train on ZJU-MoCap
python train.py dataset=zjumocap_377_mono

# Evaluate
python render.py mode=test dataset=zjumocap_377_mono

# Render novel poses
python render.py mode=predict dataset=zjumocap_377_mono
```
