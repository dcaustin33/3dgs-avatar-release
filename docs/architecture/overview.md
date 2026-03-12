# Architecture Overview

3DGS-Avatar is a system for creating animatable human avatars via deformable 3D Gaussian Splatting. Given monocular or multi-view video with SMPL body fits, it learns a set of 3D Gaussians in a canonical pose space and deforms them per-frame using skeletal (rigid) and pose-conditioned (non-rigid) deformations. The result is photorealistic, real-time renderable avatars that can be driven by novel poses.

## Tech Stack

- **Language:** Python 3.7
- **Framework:** PyTorch 1.12 with CUDA 11.6
- **Config:** Hydra + OmegaConf (YAML-based hierarchical config)
- **Body Model:** SMPL (male/female/neutral)
- **Rendering:** Differentiable Gaussian rasterization (CUDA submodule from 3DGS)
- **Spatial Queries:** simple-knn (CUDA), pytorch3d KNN
- **Neural Features:** tinycudann hash grid encoding
- **Logging:** Weights & Biases (wandb)
- **Mesh Ops:** trimesh, libigl, plyfile

## Subsystem Map

```
                    ┌─────────────────┐
                    │   Entry Points   │
                    │ train.py/render.py│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │      Scene       │
                    │  scene/__init__  │
                    │ (orchestrates    │
                    │  data + model)   │
                    └──┬──────────┬───┘
                       │          │
          ┌────────────▼──┐  ┌───▼──────────────┐
          │   Datasets     │  │ GaussianConverter │
          │ dataset/*.py   │  │ (model pipeline)  │
          └────────────────┘  └──┬───┬───┬───┬───┘
                                 │   │   │   │
              ┌──────────────────┘   │   │   └──────────────────┐
              │                      │   │                      │
     ┌────────▼───────┐  ┌──────────▼─┐ │ ┌────────────────┐   │
     │ Pose Correction │  │  Deformer  │ │ │    Texture     │   │
     │ (SMPL optimize) │  │ rigid +    │ │ │ (SH or MLP)   │   │
     └────────────────┘  │ non-rigid  │ │ └────────────────┘   │
                          └────────────┘ │                      │
                                         │              ┌───────▼───────┐
                                         │              │ GaussianModel │
                                         │              │ (3DGS points) │
                                         └──────────────┴───────────────┘
```

**Data flows top-down:** Config drives entry points, which create a `Scene`. The Scene loads datasets and initializes the `GaussianConverter` pipeline. Each training iteration: sample a frame → pose-correct SMPL params → non-rigid deform Gaussians → rigid (LBS) deform → compute colors → rasterize → compute loss → backprop.

## Subsystems

| Subsystem | Purpose | Key Entry Point |
|-----------|---------|-----------------|
| [Training & Rendering](training-rendering.md) | Training loop, evaluation, inference | `train.py`, `render.py` |
| [Scene Management](scene-management.md) | Orchestrates datasets + model pipeline | `scene/__init__.py` |
| [Gaussian Primitives](gaussian-primitives.md) | 3D Gaussian point cloud with densification | `scene/gaussian_model.py` |
| [Deformation Pipeline](deformation-pipeline.md) | Non-rigid + rigid deformation of Gaussians | `models/deformer/` |
| [Pose Correction](pose-correction.md) | Per-frame SMPL parameter optimization | `models/pose_correction/` |
| [Texture & Appearance](texture-appearance.md) | View-dependent color computation | `models/texture/texture.py` |
| [Datasets](datasets.md) | ZJU-MoCap and PeopleSnapshot data loading | `dataset/*.py` |

## Key Architectural Patterns

- **Factory pattern** — All model components (deformer, pose correction, texture) use `get_*()` factory functions dispatching on config strings. `[OBSERVED]`
- **Canonical space deformation** — Gaussians live in a star-posed (Vitruvian A-pose) canonical space and are deformed per-frame. `[OBSERVED]`
- **Composition over inheritance** — `GaussianConverter` composes pose correction, deformer, and texture as independent `nn.Module` sub-components. `[OBSERVED]`
- **Hydra config composition** — Configs are split by concern (dataset, rigid, non_rigid, texture, pose_correction, option) and composed at runtime. `[OBSERVED]`
- **Curriculum learning** — Non-rigid deformation has delay mechanisms; `HannwCondMLP` gradually introduces higher frequency bands. `[OBSERVED]`

## External Dependencies

| Dependency | Role |
|-----------|------|
| SMPL body model (`.pkl` files) | Parametric human body mesh (6890 verts, 24 joints) |
| diff-gaussian-rasterization (submodule) | CUDA differentiable Gaussian rasterizer |
| simple-knn (submodule) | CUDA nearest-neighbor distance computation |
| tinycudann | Multi-resolution hash grid encoding |
| wandb | Experiment logging and image visualization |
| pytorch3d | KNN queries for skinning and AIAP loss |
