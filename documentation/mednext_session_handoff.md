# MedNeXt architecture and nnU-Net v2 session handoff

## Objective and authorization

Build a native nnU-Net v2 architecture using MedNeXt blocks and ConvNeXt V2 features, with adaptive planning and ordinary supervised training. The comparison target is nnU-Net ResEncM, not faithful reproduction of baseline MedNeXt.

The user explicitly approved implementation with “please go ahead and implement.” Architecture repairs, the full-resolution planner, the SGD trainer and inherited AdamW trainer are implemented in the working trees. Do not ask for implementation approval again. The neighboring-repository write was also approved through tool escalation. No commits or history rewrites have been made.

Usage, configuration semantics, and profiling commands are in `documentation/mednext.md`. Implementation status below supersedes the original static review and proposed sequence later in this handoff.

## Implementation status

- Architecture uses registered encoder/decoder modules, validated resolution levels, independent channel widths and anisotropic strides, projected residual resampling, propagated GRN, selectable normalization, optional DropPath and non-reentrant checkpointing.
- Full-resolution planner reuses vanilla spacing, topology, normalization, resampling, and plan assembly. It serializes all architecture options and a provisional analytical memory estimate. Its default plans name is `MedNeXtMPlans` (M) or `MedNeXtLPlans` (L).
- Trainer names are `nnUNetTrainerMedNeXt` and `nnUNetTrainerMedNeXtAdamW`, both in `nnunetv2/training/nnUNetTrainer/nnUNetTrainerMedNeXt.py`. The latter inherits from the SGD trainer. The planner omits global `initial_lr`, allowing trainer-specific defaults; an explicit global value overrides either default.
- Constructor `architecture_overrides` applies block/capacity options before memory planning; `kernel_size` adjusts default anisotropic kernels. Planner-controlled geometry cannot be overridden there. Direct geometry edits in plans require consistent shapes and patch divisibility, validated at network reconstruction. Direct capacity edits invalidate memory estimates and should be applied through replanning.
- Added `nnunetv2/experiment_planning/profile_mednext.py` for synthetic FP16 CUDA/ROCm memory measurements with actual segmentation losses, warmup, optimizer state, and optional compilation/checkpointing. This is separate from segmentation-quality validation and never runs automatically during planning.
- New tests are in `nnunetv2/tests/test_mednext.py`, including both trainers' real training steps, checkpoint save/load, preprocessing, standard predictor reconstruction and sliding-window inference, gradients, registration, odd/anisotropic shapes, GRN stability, checkpointing with DropPath, and compiled graph capture with the eager backend.
- Final targeted run: 28 tests passed (24 MedNeXt checks plus four path regression checks). CLI planner discovery and source parsing also passed, and both repositories passed `git diff --check`. A broader run had 30 passing checks and the two unrelated discovery failures described below.
- Feature-map estimates were checked against actual convolution output counts. Memory coefficients remain uncalibrated: 12 bytes per convolution output element, 24 bytes per parameter, and a 20% target-budget reserve, with no checkpointing discount.
- PyTorch is `2.12.0+rocm7.14.0`, but no GPU or `/dev/kfd` is exposed in this session. Accelerator profiling, GPU Inductor kernels, and distributed runtime behavior remain unverified. CPU eager and compiled-eager checks are available.
- Broader regression tests found two existing failures in `test_find_objects.py`: unrelated `variants/pretrain/inference.py` imports the absent `nnUNetTrainer.UniRepLKTrainer`. MedNeXt trainer lookup and standard inference reconstruction passed. That unrelated module was not changed.
- Test artifacts and matplotlib/temp caches are directed into ignored `nnunetv2/tests/.mednext-runtime/`. Use the environment and pytest arguments in the usage guide. The active installed architecture package initially pointed to site-packages, so tests explicitly prepend the neighboring source tree to PYTHONPATH; no environment package installation was performed.

The user permits reading both repositories, tests writing only into those repositories, and Git operations that preserve history. Actual sandbox permissions still apply: the neighboring repository is currently outside writable roots and may require tool escalation. Do not rewrite history or discard unrelated changes. Do not spawn subagents unless separately authorized or required by applicable repository instructions.

## Repositories and review baseline

- nnU-Net: `/home/palpatine/alpaca/nnUNet`.
- Architecture repository: `/home/palpatine/alpaca/dynamic-network-architectures` (plural; the user's original singular path was a typo).
- Starting architecture commit: `5431f66719f8fa978bf6cb5f93a5cc2b8418a130`, titled `feat: use mednext with nnunetv2`.
- Current architecture and block files were inspected as well as that commit. Current LayerNorm implementation differs from the original commit; preserve awareness of current HEAD rather than blindly restoring the old implementation.
- Both repositories had clean working trees before this handoff was created. Recheck on resumption.
- No applicable AGENTS.md was found in the repositories during the initial search; check applicable ancestor instructions on resumption.

Relevant files:

- `../dynamic-network-architectures/dynamic_network_architectures/architectures/mednext.py`
- `../dynamic-network-architectures/dynamic_network_architectures/building_blocks/mednext.py`
- `nnunetv2/experiment_planning/experiment_planners/mednext.py`
- `nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py`
- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`
- `nnunetv2/utilities/get_network_from_plans.py`
- `nnunetv2/training/nnUNetTrainer/variants/optimizer/nnUNetTrainerAdam.py`

## Accepted architecture decisions

- Native nnU-Net behavior, with architectural parameters serialized in plans and configurable by the user.
- Only `3d_fullres` needs end-to-end support. No 2D, 3D low-resolution, or cascade implementation is required. Their omission must not break normal planner execution, discovery, preprocessing, or inference.
- No masked autoencoder training, pretraining pipeline, or changes to the main supervised training loop.
- Adapt resolution-level count, anisotropic strides and kernels, patch size, and batch size to dataset geometry and VRAM target.
- Keep capacity fixed by the selected preset unless explicitly overridden; do not silently reduce widths or block counts to fit memory.
- Initial capacity: base width 32, doubling channels by resolution level with the vanilla 3D channel cap, two blocks per encoder/decoder stage, expansion ratio 4. Include the bottleneck in a consistent encoder-level specification.
- Spatial strides and channel widths are independent. Resampling must use the next level's explicit channel count.
- Size-three kernels by default, with anisotropic adaptation. Larger size-five/seven kernels are configurable options.
- GroupNorm plus GRN by default. Channel-wise LayerNorm remains configurable.
- GRN propagates through encoder, bottleneck, decoder, downsampling, and upsampling blocks.
- Residual connections enabled for ordinary blocks and projected residual connections enabled for resampling blocks.
- Additive encoder-to-decoder skips retained.
- DropPath supported as an option, initially zero.
- Activation checkpointing supported as an optional memory-saving mode; use a consistent implementation and verify gradients.
- UpKern transfer and staged small-kernel-to-large-kernel training deferred.
- Decoder-only deep supervision, highest resolution first, matching vanilla nnU-Net scales and weighting. No bottleneck supervision implementation: user decided it has little value for their dataset.
- Inference returns only full-resolution logits when deep supervision is disabled.
- Construction must allow framework-controlled device placement, even though training is GPU-focused. Remove hard-coded CUDA allocations.

## Planner and comparison decisions

The user's comparison uses ResEncM with defaults except manually overridden spacing. They will manage comparison settings themselves. Preserve vanilla nnU-Net trainer/planner defaults except where this architecture or its explicit variant requires differences. Do not force matching ResEncM patch sizes, spacing, or model capacity.

The specialized planner must emit a valid full-resolution-only plans file and preserve normal metadata, fingerprint use, target-spacing overrides, preprocessing, CLI discovery, and saving. The default planner currently always constructs 2D and conditionally constructs low-resolution/cascade configurations, so merely rejecting 2D in the architecture will break planning. Select a minimal maintainable integration after reviewing the remaining planner code.

Provide architecture-specific feature-map/memory estimation. Vanilla U-Net VRAM constants cannot be assumed accurate for expanded depthwise blocks, optimizer state, or checkpointing. Compute patch and batch size consistently with the final architecture; do not replace patch size after inheriting a batch estimate for another architecture.

All resolved architectural choices are present in plans. Override schema and replanning semantics are documented in `documentation/mednext.md`.

## Trainers and optimizer decisions

Provide both variants through inheritance, with proposed names:

```text
nnUNetTrainer
  nnUNetTrainerMedNeXt
    nnUNetTrainerMedNeXtAdamW
```

Names are proposed, not a user-mandated API. The base architecture trainer retains vanilla SGD and scheduling. The AdamW subclass changes optimizer construction and default learning rate with minimal overrides. Preserve explicit plan overrides. Reuse the nnU-Net v2 polynomial scheduler rather than copying v1's manual scheduler mechanism.

Verified upstream MedNeXt optimizer recipe:

- AdamW.
- Initial learning rate `1e-3`.
- Weight decay `3e-5`, inherited from its base trainer.
- Epsilon `1e-4`.
- Default AdamW betas `(0.9, 0.999)`.
- Polynomial learning-rate decay with exponent `0.9`, inherited from its v1 trainer.
- Upstream setting `lr_scheduler = None` does not mean constant learning rate: its parent updates the LR each epoch.

Keep other training hyperparameters, augmentation, losses, and training behavior inherited from the local vanilla trainer. The user allows narrow trainer-specific optimizer, schedule, and deep-supervision integration overrides. The local vanilla trainer already reads some global plan hyperparameters, including initial LR and epoch count; inspect before designing override precedence.

## Confirmed static review findings

These are code-inspection findings, not results of execution. Line references below describe the reviewed files and can shift after edits.

1. Architecture block collections use ordinary lists instead of registered module containers. Encoder, decoder, resampling, and output-head parameters are excluded from normal parameter discovery, state dictionaries, mode changes, and device movement.
2. Stage semantics conflict. Construction near architecture line 66 uses `zip(channels, strides[1:], kernel_sizes)` while forward near line 181 uses `range(n_stages)`. Four resolution levels with four strides create three transition pairs, but forward accesses four.
3. Scalar block count expands to `n_stages` entries near line 45, but decoder indexing starts at `2 * n_stages` and bottleneck indexing uses `n_stages`. For four levels, four entries are indexed at eight and four. Even a nine-entry list leaves unused indices under the current construction loop.
4. Bottleneck channels are `channels[-1]`, but the extra output head near line 139 expects `channels[-1] * strides[-1][0]`. A 256-channel bottleneck can be sent to a head expecting 512 channels.
5. Down/up channel counts use `stride[0] * channel` rather than explicit neighboring channel widths. Anisotropic stride `(1,2,2)` therefore fails to increase channels even when the next stage expects twice as many.
6. Resampling constructors do not receive the planned strides; padding assumes stride two on every spatial axis. Fix anisotropic stride and output alignment together.
7. Hard-coded CUDA allocation occurs throughout blocks and heads, conflicting with CPU planner construction and framework device placement.
8. Residual upsampling is incorrect when enabled. UpBlock forwards `do_res=True` to the base block, whose forward adds the original input to the spatially upsampled/channel-compressed result before padding or projection. Example: `[B,128,8,8,8]` plus `[B,64,15,15,15]`. Disable the base identity residual for resampling; use the separate projected residual. IMPORTANT: an earlier assistant claim that positional arguments were misaligned was corrected. The arguments do match; the problem is enabling the base identity residual.
9. Current channel-first LayerNorm broadcasting is hard-coded to 3D. Since only 3D is required, broader dimension support is not an acceptance requirement; still make the supported contract explicit.
10. GRN formula and placement are consistent with upstream, but the architecture omits passing GRN to downsampling blocks. Verify numerical behavior under mixed precision.
11. Existing planner uses incompatible constructor names (`features_per_stage` versus `channels`), lists missing kwargs in `_kw_requires_import`, and supplies malformed/inconsistent stage/kernel/stride settings. The inherited planner also attempts to instantiate this architecture with vanilla kwargs.
12. Architecture lacks `compute_conv_feature_map_size`, called by the default planner's static estimator.
13. Deep-supervision convention differs from vanilla. Four resolution levels have target scales `1, 1/2, 1/4`; the bottleneck `1/8` is excluded. Follow decoder-only outputs rather than implementing extra targets.
14. Local trainer deep-supervision toggle looks for `mod.decoder.deep_supervision`; without a decoder it sets `self.deep_supervision` on the trainer instead of the network. Prefer a compatible decoder interface or narrowly override the toggle, including DDP and compiled wrappers.
15. Remove/reconsider repeated `torch.cuda.empty_cache()` calls inside forward and the legacy dummy-parameter checkpoint workaround as part of a coherent memory/checkpointing implementation.

Recommended topology contract: `n_stages` is the number of resolution levels including bottleneck; there are `n_stages - 1` down/up transitions. Use separate encoder and decoder block-count lists and explicit per-level channels. Validate all list lengths and shapes before constructing layers.

## Testing and memory profiling

Development environment uses ROCm with the correct PyTorch version according to the user. Do not assume NVIDIA CUDA is present. PyTorch ROCm intentionally exposes devices and memory APIs through `torch.cuda`; inspect `torch.version.hip` and actual availability before testing.

Calibration means local synthetic forward/backward/optimizer-step testing and peak-memory profiling, not dataset training, segmentation validation, or accuracy measurement. Include optimizer-state allocation and warmup, and cover SGD and AdamW. ROCm measurements validate local behavior but do not establish NVIDIA peak memory because backend workspaces and execution choices differ. Keep planner estimates and measured results clearly distinguished. Avoid making GPU profiling a mandatory requirement every time the planner runs.

Required validation after implementation:

- Constructor validation, stage counts, isotropic and anisotropic shapes.
- All intended parameters registered, optimized, moved, and checkpointed.
- Gradients and finite mixed-precision training steps where supported.
- Residual up/down shapes and GRN propagation.
- Checkpointing on/off correctness.
- Deep-supervision output/target count, ordering, resolution, and inference toggle.
- Plans serialization and network reconstruction using standard nnU-Net utilities.
- Full-resolution-only planning and preprocessing configuration discovery.
- Checkpoint save/load and inference reconstruction for both trainer classes.
- Synthetic memory profiling with backend, precision, optimizer, and checkpoint settings recorded.

The static review preceded runtime testing; current results and profiling limits are recorded in the implementation status above. Exact dataset details and final training GPU remain unspecified; do not repeatedly ask for comparison settings the user intends to manage.

## Implementation sequence

1. Recheck repository instructions, status, environment, and current code.
2. Repair reusable blocks and implement the validated encoder/decoder architecture contract.
3. Implement memory accounting and the full-resolution planner, with a clear override schema.
4. Add the base SGD trainer and small inherited AdamW trainer.
5. Run structural and integration tests, then available ROCm profiling; fix failures.
6. Document planning, training, inference commands, configurable settings, and memory-estimate limitations. Report actual checks and remaining platform gaps.

These steps have been carried out to the extent supported by local device access; GPU profiling remains outstanding on an accelerator-enabled session.

## Sources

- MedNeXt repository: https://github.com/MIC-DKFZ/MedNeXt
- MedNeXt blocks: https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/network_architecture/mednextv1/blocks.py
- MedNeXt optimizer and variants: https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/training/network_training/MedNeXt/nnUNetTrainerV2_MedNeXt.py
- MedNeXt parent scheduling: https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/training/network_training/nnUNetTrainerV2.py
- MedNeXt parent weight decay: https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/training/network_training/nnUNetTrainer.py
- ConvNeXt V2 paper: https://arxiv.org/abs/2301.00808
- ConvNeXt V2 block: https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
- PyTorch ROCm semantics: https://docs.pytorch.org/docs/2.14/notes/hip.html

The grill-me skill was used for the design discussion. Continue resolving genuine design dependencies, but do not re-ask already accepted decisions above.

Current planner entry points are `MedNeXtPlannerM` and `MedNeXtPlannerL`; the unreleased legacy `MedNeXtPlanner` entry point was removed. Both use checkpoint-aware provisional peak estimation. See mednext.md for current recipes and calibration assumptions.
