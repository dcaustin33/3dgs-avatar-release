# Scene Management

## Purpose

`Scene` is the central coordinator that connects datasets with the model pipeline. It owns the `GaussianModel` (point cloud), the `GaussianConverter` (deformation + appearance), and train/test datasets.

## Key Abstractions

- `Scene` class in `scene/__init__.py` — Orchestrator
- `GaussianConverter` in `models/gaussian_converter.py` — Model pipeline (pose correction → deformer → texture)

## Data Flow

1. `Scene.__init__()`:
   - Loads train/test datasets via `load_dataset()`
   - Extracts `metadata` from training dataset (canonical SMPL mesh, AABB, skinning weights)
   - Initializes `GaussianModel` from point cloud (SMPL mesh vertices)
   - Creates `GaussianConverter` with metadata
2. `Scene.convert_gaussians(camera, iteration)`:
   - Delegates to `GaussianConverter.forward()` which runs the full pipeline
   - Returns deformed Gaussian model + regularization losses + precomputed colors
3. `Scene.optimize(iteration)`:
   - Steps both the Gaussian optimizer (with delay) and the converter optimizer

## GaussianConverter Pipeline

`GaussianConverter.forward(gaussians, camera, iteration, compute_loss)`:

1. **Pose Correction** → adjusts SMPL parameters, updates camera with corrected bone transforms
2. **Pose Augmentation** (training only) → adds noise to joint rotations
3. **Non-Rigid Deformation** → MLP-based position/scale/rotation offsets conditioned on pose
4. **Rigid Deformation** → LBS using learned skinning weights + bone transforms
5. **Texture** → computes RGB colors (SH evaluation or MLP)

Returns: deformed `GaussianModel`, loss dict, color tensor.

## Key Files

| File | Role |
|------|------|
| `scene/__init__.py` | `Scene` class — data + model orchestrator |
| `models/__init__.py` | Exports `GaussianConverter` |
| `models/gaussian_converter.py` | Pipeline composition + optimizer management |

## Non-obvious Patterns

- **Metadata sharing**: The `metadata` dict from the training dataset is passed to all model components, providing canonical mesh, AABB, skinning weights, etc. `[OBSERVED]`
- **Separate optimizers**: Gaussians use Adam with per-parameter-group learning rates; converter uses a separate Adam with component-specific LR groups and exponential scheduler. `[OBSERVED]`
- **Camera copy pattern**: `GaussianConverter` copies the camera object before modifying it (pose correction updates), preserving the original. `[OBSERVED]`
