# Tversky and cross entropy training

`nnUNetTrainerTverskyLoss` replaces the Dice term with soft Tversky while retaining the standard cross-entropy term, optimizer, learning-rate schedule, deep supervision, architecture, preprocessing, and inference behavior. It works with existing ResEncM plans; no new planner or preprocessing run is required for a loss-only experiment.

## Defaults and meaning

For each foreground class, the overlap score is:

```text
T = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)
alpha = 0.3
beta = 1 - alpha = 0.7
epsilon = smooth / 2 = 5e-6
```

Alpha weights false positives, and beta weights false negatives. The default puts more weight on false negatives and encourages recall. The same alpha applies to every foreground class. Class scores are averaged equally, as with vanilla Dice; optional CE class weights affect only the CE term.

The implemented loss is `CE - mean(T)`, following nnU-Net's negative-Dice convention. Using `CE + 1 - mean(T)` would add a constant and produce identical gradients. Both terms have coefficient one. Alpha `0.5` recovers the local vanilla Dice score, including its smoothing convention. Alpha must be finite and between zero and one, inclusive. Beta is derived and cannot be set independently.

CE is unweighted by default. Different weights can be supplied for every foreground class and for background. These scale the CE contribution of voxels according to their true class; they do not independently set FP and FN penalties for each class in a multiclass softmax. Tversky supplies the FP/FN asymmetry.

## Plans configuration

Add a `loss` object inside the selected configuration in your plans JSON, usually:

```text
$nnUNet_preprocessed/DatasetXXX_NAME/nnUNetResEncUNetMPlans.json
configurations.3d_fullres.loss
```

The following settings are equivalent to omitting the entire `loss` object:

```json
"loss": {
  "tversky_alpha": 0.3,
  "ce_class_weights": null
}
```

To weight individual classes, use their names from `dataset.json`:

```json
"loss": {
  "tversky_alpha": 0.3,
  "ce_class_weights": {
    "background": 1.0,
    "nerve": 3.0,
    "muscle": 0.5
  }
}
```

Replace these example class names with your dataset's names. Missing dictionary entries receive weight one. You can also use label-ID strings, such as `{"1": 3.0, "2": 0.5}`, or a complete list in class-ID order:

```json
"ce_class_weights": [1.0, 3.0, 0.5]
```

For ordinary multiclass segmentation, the list includes background at index zero and every foreground class. Do not include the ignore label. Lists must have exactly as many entries as output classes. Weights must be finite and strictly positive; use nnU-Net's ignore-label mechanism for voxels that should be excluded. Unknown names, duplicate name/ID aliases for one output, misspelled loss settings, and independently specified beta are rejected.

Softmax CE uses PyTorch's weighted-mean reduction, so multiplying every class weight by the same constant leaves that CE loss unchanged. Relative weights matter. A larger foreground weight increases the penalty for incorrect predictions at voxels of that class; the effect on other classes is coupled through the softmax.

Configuration inheritance works through the normal plans manager. Settings belong to the resolved configuration's `loss` object, not the top level of plans. Keep a saved copy of the exact plans used for each experiment. Use a separate plans identifier for multiple alpha/weight experiments to avoid overwriting checkpoints under the same trainer/configuration/fold path.

## Training and inference

With the ResEncM dataset already preprocessed:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainerTverskyLoss
```

The trainer uses vanilla SGD, the ordinary polynomial schedule, and any existing supported plan overrides. It adds no architecture changes and does not require the new MedNeXt implementation.

To disable mirror augmentation and mirror-based inference ensembling, use the inherited variant:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD \
  -p nnUNetResEncUNetMPlans -tr nnUNetTrainerTverskyLossNoMirroring
```

It uses the same loss settings and retains the standard rotation augmentation. Checkpoints record that inference mirroring is disabled. Use `-tr nnUNetTrainerTverskyLossNoMirroring` in the prediction command for this variant.

For inference:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID \
  -c 3d_fullres -p nnUNetResEncUNetMPlans \
  -tr nnUNetTrainerTverskyLoss -f FOLD
```

If you use a separate plans identifier, substitute it in both commands. The trained model folder preserves its plans and trainer identity for checkpoint reconstruction. The loss is used only during training and loss-based validation; final prediction and Dice-based evaluation metrics use standard nnU-Net behavior. A loss-only change leaves network tensor shapes compatible with ordinary ResEnc, but keep optimizer checkpoints and experiment outputs distinct.

## Framework behavior

The implementation preserves foreground-only Tversky for mutually exclusive classes, optional batch aggregation, DDP aggregation of overlap statistics, ignore-label masking, and deep-supervision weights. It uses FP32 reductions without allocating full-sized FP/FN/TN tensors. A single supervision head receives weight one instead of an all-zero weight vector.

Region-based nnU-Net configurations use sigmoid Tversky plus BCE, as the vanilla trainer uses Dice plus BCE. In that mode, CE/BCE weights refer to region output channels in dataset order, excluding background; names may reference region names. A region's weight multiplies both its positive and negative BCE terms. It is not a BCE `pos_weight` and adds no additional FP/FN asymmetry. The final ignore-mask channel is never given a class weight.

## Validation

Tests cover the direct multiclass Tversky definition and gradients, Dice equivalence at alpha `0.5`, FP/FN weighting, omitted classes, fully ignored targets, reduced-precision inputs, per-class CE weights, region BCE weights, configuration validation, deep-supervision/DDP weight selection, and graph capture using the eager compilation backend. A synthetic ResEncM integration test exercises a real optimizer step, validation loss, checkpoint save/resume, trainer discovery, and standard sliding-window prediction. Accelerator kernels and distributed collectives require separate hardware-enabled validation.

To run the tests while keeping generated files in the repository:

```bash
mkdir -p nnunetv2/tests/.tversky-runtime
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD" \
MPLCONFIGDIR="$PWD/nnunetv2/tests/.tversky-runtime/matplotlib" \
TMPDIR="$PWD/nnunetv2/tests/.tversky-runtime" \
python -m pytest nnunetv2/tests/test_tversky.py \
  --basetemp=nnunetv2/tests/.tversky-runtime/pytest -p no:cacheprovider -q
```
