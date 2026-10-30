# MedNeXt with nnU-Net v2

`MedNeXtPlanner` creates a native `3d_fullres` configuration. `nnUNetTrainerMedNeXt` uses vanilla SGD; `nnUNetTrainerMedNeXtAdamW` inherits that trainer and switches to the MedNeXt AdamW recipe. Both use the same plans and ordinary nnU-Net training and inference commands.

## Install the source implementations

The architecture changes live in the neighboring `dynamic-network-architectures` repository. Install both edited repositories into your training environment; the released architecture package does not contain this implementation:

```bash
python -m pip install -e ../dynamic-network-architectures
python -m pip install -e .
```

For local checks without changing the environment's installed packages, prepend both source repositories to `PYTHONPATH`. Verify the active architecture source:

```bash
python -c 'import dynamic_network_architectures.architectures.mednext as m; print(m.__file__)'
```

Old experimental MedNeXt plans and checkpoints are not compatible with the repaired stage contract and module registration. Generate new plans and start a new training run.

## Plan and preprocess

With the standard nnU-Net environment paths configured:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl MedNeXtPlanner -c 3d_fullres
```

The default plans identifier is `MedNeXtPlans`. The planner accepts the normal `-gpu_memory_target`, `-overwrite_target_spacing`, and `-overwrite_plans_name` arguments. Target spacing, normalization, resampling, transpose, minimum batch size, dataset batch-coverage cap, and minimum bottleneck edge length follow vanilla planner defaults. It emits no 2D, low-resolution, or cascade configuration. Normal preprocessing skips missing configurations if invoked with its default configuration list.

The default encoder recipe is widths `[32, 64, 128, 384, 512, 512]`, block counts `[2, 3, 4, 6, 6, 6]`, and expansion ratios `[4, 4, 4, 8, 8, 8]`. Shorter networks use the prefix; further stages repeat the last value. Decoder stages have two blocks and mirror the destination encoder expansion. Checkpointing is enabled. With six stages, one input channel and five outputs, this has approximately 96.67M parameters, compared with 101.68M for the reference ResEncM plan; parameter count is not a planning target. Stage count, anisotropic strides, patch size, and batch size adapt to spacing, median shape, and requested memory. The memory model remains provisional and does not account for checkpointing savings; measure on the training GPU before relying on its estimates.

The planner uses a provisional analytical memory model: convolution-output counts include expanded activations, resampling projections, and segmentation heads; parameter storage allows for AdamW states and optimizer temporaries. It reserves 20% of the target budget. These coefficients are uncalibrated and do not guarantee that a plan fits a particular accelerator. Checkpointing receives no assumed memory discount. The resolved estimate and coefficients are recorded in each configuration's `memory_estimate`.

## Train and predict

Vanilla SGD:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -p MedNeXtPlans -tr nnUNetTrainerMedNeXt
```

MedNeXt AdamW:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -p MedNeXtPlans -tr nnUNetTrainerMedNeXtAdamW
```

| Setting | SGD trainer | AdamW trainer |
| --- | --- | --- |
| Initial learning rate | `1e-2` | `1e-3` |
| Weight decay | `3e-5` | `3e-5` |
| Momentum | `0.99`, Nesterov | Betas `(0.9, 0.999)` |
| Epsilon | Not applicable | `1e-4` |
| Schedule | Polynomial, exponent `0.9` | Polynomial, exponent `0.9` |
| Epochs | `1000` | `1000` |

The planner intentionally omits a global `initial_lr`; each trainer supplies its default. Setting `initial_lr` explicitly at the top level of the plans overrides either trainer's default. Existing local vanilla overrides such as `num_epochs`, `lr_exponent`, and `save_every` remain inherited. Augmentation, loss, sampling, clipping, validation, and checkpoint handling remain inherited. Decoder-only supervision uses vanilla weighting; a two-level network with one decoder head receives weight one to avoid vanilla's all-zero weighting edge case.

Prediction uses the trainer recorded in the checkpoint:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres \
  -p MedNeXtPlans -tr nnUNetTrainerMedNeXtAdamW -f FOLD
```

Use `nnUNetTrainerMedNeXt` instead when predicting an SGD run. Standard prediction reconstructs the complete model from plans, loads all registered parameters, and disables auxiliary outputs.

## Architecture configuration

All resolved settings are stored under `configurations.3d_fullres.architecture.arch_kwargs`:

| Setting | Meaning |
| --- | --- |
| `n_stages` | Resolution levels including the bottleneck; at least two |
| `features_per_stage` | Explicit widths per level; scalar input doubles width per level |
| `strides` | One 3D tuple per level; first identity, later values one or two |
| `kernel_sizes` | Odd kernels per level; default size three with anisotropic size-one axes |
| `n_conv_per_stage` | Encoder block counts, full resolution to bottleneck |
| `n_conv_per_stage_decoder` | Decoder block counts, bottleneck to full resolution |
| `expansion_ratio` | Encoder expansion ratios, full resolution to bottleneck |
| `expansion_ratio_decoder` | Decoder expansion ratios, bottleneck to full resolution; null mirrors encoder ratios |
| `norm_type` | `group` by default; `layer` selects channel-wise LayerNorm |
| `grn` | True by default, applied in ordinary and resampling blocks |
| `do_res` | Identity residuals in ordinary blocks; true |
| `do_res_up_down` | Projected residuals in resampling blocks; true |
| `drop_path_rate` | Per-block residual DropPath probability; zero |
| `outside_block_checkpointing` | Non-reentrant activation checkpointing; true in planner defaults |

Auxiliary heads exist at decoder resolutions, never the bottleneck. GroupNorm uses one group per channel. GRN reduces over spatial dimensions in FP32 under mixed precision. Additive skip fusion preserves the MedNeXt structure. Transposed depthwise convolutions and their residual projections respect anisotropic strides.

To change capacity or kernels before memory planning, use the planner's Python interface (or a small planner subclass for CLI discovery):

```python
from nnunetv2.experiment_planning.experiment_planners.mednext import MedNeXtPlanner

planner = MedNeXtPlanner(
    DATASET_ID,
    plans_name='MedNeXtCustomPlans',
    gpu_memory_target_in_gb=24,
    kernel_size=5,
    architecture_overrides={
        'expansion_ratio': 4,
        'n_conv_per_stage': 2,
        'n_conv_per_stage_decoder': 2,
        'norm_type': 'group',
        'grn': True,
        'drop_path_rate': 0.1,
        'outside_block_checkpointing': True,
    },
)
planner.plan_experiment()
```

Scalar counts/ratios adapt to the derived stage count. Explicit lists must match the derived stage count (decoder lists have one fewer entry). The planner controls stage count, strides, convolution dimensionality, and supervision mode. Its constructor rejects overrides of those fields. Plan configuration edits remain possible, but topology edits require consistent list lengths and patch divisibility; the trainer rejects incompatible patch geometry. Changing capacity, kernels, or geometry in an existing JSON invalidates its memory estimate: regenerate plans with the options applied before memory planning. A new plans name avoids mixing preprocessing and checkpoints across experiments. Spacing changes require preprocessing again.

No UpKern transfer, masked pretraining, 2D model, cascade, or bottleneck supervision is included.

## Synthetic memory profiling

On a machine with an accessible CUDA or ROCm device:

```bash
python -m nnunetv2.experiment_planning.profile_mednext \
  --plans "$nnUNet_preprocessed/DatasetXXX_NAME/MedNeXtPlans.json" \
  --optimizer adamw --warmup 3 --steps 3 --output mednext-memory-adamw.json
```

Repeat with `--optimizer sgd`. `--batch-size` can reduce the planned batch size for diagnosis; `--checkpointing` / `--no-checkpointing` override that setting for the measurement. `--compile` includes compilation and its memory behavior. Without it, profiling runs eagerly; match the actual training mode when comparing results.

This runs synthetic segmentation losses and scaled FP16 forward/backward/optimizer steps, including gradient clipping. Warmup allocates optimizer state before peak measurement; increase warmup if GradScaler initially skips steps. Output includes device/backend versions, optimizer, precision, checkpointing/compilation settings, peak allocated/reserved memory, and the planner estimate. It does not evaluate segmentation quality or automatically alter plans. ROCm measurements do not establish NVIDIA CUDA memory use; repeat on the training hardware. If a configuration runs out of memory, reduce the planning memory target and replan, or measure a smaller batch before changing the plan.

## Local checks

From the nnU-Net repository, keep test artifacts inside its test directory:

```bash
mkdir -p nnunetv2/tests/.mednext-runtime
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$(realpath ../dynamic-network-architectures):$PWD" \
MPLCONFIGDIR="$PWD/nnunetv2/tests/.mednext-runtime/matplotlib" \
TMPDIR="$PWD/nnunetv2/tests/.mednext-runtime" \
python -m pytest nnunetv2/tests/test_mednext.py \
  --basetemp=nnunetv2/tests/.mednext-runtime/pytest -p no:cacheprovider -q
```

The checks cover module registration, anisotropic and odd shapes, GRN, gradients, checkpointing with DropPath, compiled graph capture using the eager backend, feature-map counts against convolution execution, planning, preprocessing, both optimizer recipes, checkpoint resumption, and standard sliding-window prediction. Eager-backend graph capture does not validate GPU Inductor kernels or distributed execution.

The development PyTorch build is `2.12.0+rocm7.14.0`; this session exposes neither `/dev/kfd` nor a GPU through PyTorch. Accelerator memory profiling and CUDA/ROCm kernel testing require a session with device access.
