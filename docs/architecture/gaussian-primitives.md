# Gaussian Primitives

## Purpose

Manages the learnable 3D Gaussian point cloud — the core representation. Each Gaussian has position, color features (SH or learned), scale, rotation, and opacity. Supports adaptive densification (clone, split, prune) during training.

## Key Abstractions

- `GaussianModel` class in `scene/gaussian_model.py` — Learnable Gaussian point cloud
- Initialized from SMPL mesh point cloud, then optimized

## Parameters Per Gaussian

| Parameter | Shape | Activation | Purpose |
|-----------|-------|-----------|---------|
| `_xyz` | (N, 3) | identity | 3D position |
| `_features_dc` | (N, 1, C) | — | DC color/feature component |
| `_features_rest` | (N, F-1, C) | — | Higher-order SH or feature components |
| `_scaling` | (N, 3) | exp | Per-axis scale |
| `_rotation` | (N, 4) | normalize | Quaternion orientation |
| `_opacity` | (N, 1) | sigmoid | Opacity |

Where C=3 for SH mode, C=1 for feature mode. F = (sh_degree+1)^2 for SH.

## Densification Strategy

Runs between `densify_from_iter` and `densify_until_iter` every `densification_interval` steps:

1. **Clone**: Points with large view-space gradient but small scale → duplicate
2. **Split**: Points with large view-space gradient AND large scale → split into N=2 smaller Gaussians
3. **Prune**: Remove low-opacity or too-large Gaussians
4. **Opacity reset**: Every `opacity_reset_interval` iterations, reset all opacities to low value

## Key Files

| File | Role |
|------|------|
| `scene/gaussian_model.py` | `GaussianModel` — parameters, optimizer, densification, PLY I/O |
| `scene/cameras.py` | `Camera` — data container for per-frame camera + SMPL data |

## Non-obvious Patterns

- **Two color modes**: `use_sh=True` uses spherical harmonics (view-dependent); `use_sh=False` uses flat learned features decoded by a texture MLP. `[OBSERVED]`
- **Clone preserves optimizer state**: When densifying, new Gaussians get zero momentum in Adam. `[OBSERVED]`
- **`non_rigid_feature`**: The non-rigid deformer can attach extra per-Gaussian features used by the texture network. `[OBSERVED]`
