"""Tversky + CE/BCE for standard nnU-Net plans, including ResEnc presets."""
from copy import deepcopy

import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.tversky import MemoryEfficientSoftTverskyLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def resolve_ce_class_weights(value, label_manager):
    """Resolve a list in output-channel order or a mapping of label names/IDs.

    For softmax, include background. For region-based BCE, only region output
    channels are present. Unspecified dictionary entries default to one.
    """
    if value is None:
        return None
    count = label_manager.num_segmentation_heads
    if isinstance(value, dict):
        weights = [1.] * count
        aliases = {str(i): i for i in range(count)}
        for name, label in label_manager.label_dict.items():
            if name in ('background', 'ignore'):
                if name == 'background' and not label_manager.has_regions:
                    aliases[name] = 0
                continue
            if label_manager.has_regions:
                label = tuple(label) if isinstance(label, list) else label
                aliases[name] = label_manager.foreground_regions.index(label)
            else:
                aliases[name] = int(label)
        assigned = set()
        for name, weight in value.items():
            key = str(name)
            if key not in aliases:
                raise ValueError(f'Unknown CE class {name!r}; use label names or output channel indices')
            index = aliases[key]
            if index in assigned:
                raise ValueError(f'CE weight for output channel {index} was specified more than once')
            assigned.add(index)
            weights[index] = weight
    elif isinstance(value, (list, tuple)):
        weights = value
    else:
        raise ValueError('ce_class_weights must be null, a list, or a dictionary of class weights')
    if len(weights) != count:
        raise ValueError(f'ce_class_weights needs {count} entries in output channel order')
    try:
        weights = torch.tensor(weights, dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('CE class weights must be numbers') from error
    if weights.ndim != 1 or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('CE class weights must be finite and strictly positive')
    return weights


class nnUNetTrainerTverskyLoss(nnUNetTrainer):
    """Set configurations.<configuration>.loss in plans; defaults need no edits.

    Supported settings: tversky_alpha (default 0.3), ce_class_weights (default
    null/unweighted). Beta is always 1-alpha. All optimizer defaults are inherited.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        plans = deepcopy(plans)
        plans.setdefault('continue_training', False)
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _build_loss(self):
        settings = self.configuration_manager.configuration.get('loss', {})
        if not isinstance(settings, dict):
            raise ValueError('Configuration loss settings must be a dictionary')
        unknown = settings.keys() - {'tversky_alpha', 'ce_class_weights'}
        if unknown:
            raise ValueError(f'Unknown Tversky loss settings: {sorted(unknown)}; beta is always 1-alpha')
        alpha = settings.get('tversky_alpha', 0.3)
        ce_weights = resolve_ce_class_weights(settings.get('ce_class_weights'), self.label_manager)
        if ce_weights is not None:
            ce_weights = ce_weights.to(self.device)
        overlap_kwargs = {'batch_dice': self.configuration_manager.batch_dice,
                          'do_bg': self.label_manager.has_regions, 'smooth': 1e-5,
                          'ddp': self.is_ddp, 'alpha': alpha}
        if self.label_manager.has_regions:
            # BCEWithLogitsLoss broadcasts weights from the trailing dimensions.
            # Weight both positive and negative terms of each region equally;
            # this is distinct from BCE pos_weight and does not add FP/FN bias.
            bce_kwargs = {} if ce_weights is None else {
                'weight': ce_weights.reshape(1, -1, *([1] * len(self.configuration_manager.patch_size)))}
            loss = DC_and_BCE_loss(bce_kwargs, overlap_kwargs,
                                   use_ignore_label=self.label_manager.has_ignore_label,
                                   dice_class=MemoryEfficientSoftTverskyLoss)
        else:
            ce_kwargs = {} if ce_weights is None else {'weight': ce_weights}
            loss = DC_and_CE_loss(overlap_kwargs, ce_kwargs, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftTverskyLoss)
        self.print_to_log_file(f'Tversky alpha (FP): {loss.dc.alpha}; beta (FN): {loss.dc.beta}; '
                               f'CE class weights: {None if ce_weights is None else ce_weights.flatten().tolist()}')
        compiled = self._do_i_compile()
        if compiled:
            loss.dc = torch.compile(loss.dc)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / 2 ** i for i in range(len(scales))], dtype=float)
            if len(weights) > 1:
                weights[-1] = 1e-6 if self.is_ddp and not compiled else 0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerTverskyLossNoMirroring(nnUNetTrainerTverskyLoss):
    """Tversky + CE/BCE with mirror augmentation and inference mirroring disabled."""
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, _ = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, None
