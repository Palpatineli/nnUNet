"""Full-resolution MedNeXt planning with an explicit, provisional memory model."""
from copy import deepcopy
from math import prod

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from dynamic_network_architectures.architectures.mednext import MedNeXt

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.utils import copy_no_perms


class MedNeXtPlanner(ExperimentPlanner):
    """Only creates 3d_fullres; capacity is fixed while spatial topology adapts.

    architecture_overrides are applied BEFORE patch/batch planning. Geometry
    (n_stages/strides) is derived from spacing; edit these only in a separately
    validated custom configuration. Per-stage lists must match the derived depth.
    Memory constants are conservative heuristics, not CUDA/ROCm calibration.
    """
    def __init__(self, dataset_name_or_id, gpu_memory_target_in_gb=8,
                 preprocessor_name='DefaultPreprocessor', plans_name='MedNeXtPlans',
                 overwrite_target_spacing=None, suppress_transpose=False,
                 architecture_overrides=None, kernel_size=3):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name,
                         plans_name, overwrite_target_spacing, suppress_transpose)
        if not np.isfinite(gpu_memory_target_in_gb) or gpu_memory_target_in_gb <= 0:
            raise ValueError('gpu_memory_target_in_gb must be positive and finite')
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError('kernel_size must be a positive odd integer')
        self.UNet_class = MedNeXt
        self.architecture_overrides = deepcopy(architecture_overrides or {})
        forbidden = {'n_stages', 'strides', 'conv_op', 'input_channels', 'num_classes', 'deep_supervision'}
        if forbidden.intersection(self.architecture_overrides):
            raise ValueError(f'Planner controls {sorted(forbidden)}; override capacity/block options instead')
        self.kernel_size = kernel_size
        self.memory_safety_fraction = 0.8
        self.activation_bytes_per_element = 12
        # Covers FP32 parameters, gradients, AdamW states and temporary optimizer tensors.
        # Shared plans support both optimizers; SGD may leave more headroom.
        self.parameter_bytes_per_element = 24

    def _architecture(self, strides, kernels):
        n_stages = len(strides)
        def recipe(values):
            return [values[min(i, len(values) - 1)] for i in range(n_stages)]
        kwargs = {
            'n_stages': n_stages,
            'features_per_stage': recipe([32, 64, 128, 384, 512, 512]),
            'conv_op': 'torch.nn.Conv3d',
            'kernel_sizes': [[self.kernel_size if k > 1 else 1 for k in stage] for stage in kernels],
            'strides': [list(stage) for stage in strides],
            'n_conv_per_stage': recipe([2, 3, 4, 6, 6, 6]),
            'n_conv_per_stage_decoder': [2] * (n_stages - 1),
            'expansion_ratio': recipe([4, 4, 4, 8, 8, 8]),
            'expansion_ratio_decoder': None,
            'do_res': True,
            'do_res_up_down': True,
            'norm_type': 'group',
            'grn': True,
            'drop_path_rate': 0.,
            'outside_block_checkpointing': True,
        }
        kwargs.update(deepcopy(self.architecture_overrides))
        return {'network_class_name': MedNeXt.__module__ + '.MedNeXt',
                'arch_kwargs': kwargs, '_kw_requires_import': ['conv_op']}

    def estimate_memory(self, patch_size, architecture):
        """Return uncalibrated bytes for fixed state and one sample's activations.

        Checkpointing deliberately receives no speculative memory discount.
        Meta construction avoids allocating a potentially large CPU/GPU model.
        """
        label_manager = LabelManager(self.dataset_json['labels'], self.dataset_json.get('regions_class_order'))
        n_channels = len(self.dataset_json.get('channel_names', self.dataset_json.get('modality', {})))
        with torch.device('meta'):
            net = get_network_from_plans(
                architecture['network_class_name'], architecture['arch_kwargs'],
                architecture['_kw_requires_import'], n_channels, label_manager.num_segmentation_heads,
                allow_init=False, deep_supervision=True)
        feature_elements = net.compute_conv_feature_map_size(patch_size)
        parameters = sum(p.numel() for p in net.parameters())
        fixed = parameters * self.parameter_bytes_per_element
        per_sample = (feature_elements * self.activation_bytes_per_element +
                      prod(patch_size) * (n_channels * 4 + label_manager.num_segmentation_heads * 8))
        return int(fixed), int(per_sample)

    def get_plans_for_configuration(self, spacing, median_shape, data_identifier,
                                    approximate_n_voxels_dataset, _cache):
        spacing, median_shape = np.asarray(spacing, dtype=float), np.asarray(median_shape, dtype=float)
        if (spacing.shape != (3,) or median_shape.shape != (3,) or
                not np.all(np.isfinite(spacing)) or not np.all(np.isfinite(median_shape)) or
                np.any(spacing <= 0) or np.any(median_shape < 1)):
            raise ValueError('MedNeXtPlanner requires positive finite 3D spacing and shape')
        inverse_spacing = 1 / spacing
        initial = np.minimum(np.round(inverse_spacing * (256 ** 3 / prod(inverse_spacing)) ** (1 / 3)),
                             median_shape).astype(int)
        initial = np.maximum(initial, 1)
        budget = self.UNet_vram_target_GB * 1024 ** 3 * self.memory_safety_fraction
        while True:
            _, strides, kernels, patch, divisible = get_pool_and_conv_props(
                spacing, initial, self.UNet_featuremap_min_edge_length, 999999)
            if len(strides) < 2:
                raise RuntimeError('Dataset shape or memory budget is too small for a MedNeXt downsampling stage')
            architecture = self._architecture(strides, kernels)
            # Lists supplied as overrides are validated by network construction.
            fixed, per_sample = self.estimate_memory(patch, architecture)
            if fixed + self.UNet_min_batch_size * per_sample <= budget:
                break
            # Follow vanilla's aspect-ratio reduction, accounting for topology padding.
            reduced = False
            for axis in np.argsort(np.asarray(patch) / median_shape)[::-1]:
                candidate = np.asarray(patch).copy()
                for decrement in range(1, int(patch[axis])):
                    candidate[axis] = patch[axis] - decrement
                    _, new_strides, _, new_patch, _ = get_pool_and_conv_props(
                        spacing, candidate, self.UNet_featuremap_min_edge_length, 999999)
                    if prod(new_patch) < prod(patch) and len(new_strides) >= 2:
                        initial = candidate
                        reduced = True
                        break
                if reduced:
                    break
            if not reduced:
                raise RuntimeError('MedNeXt cannot fit minimum batch size in the requested memory budget')
        batch_size = int((budget - fixed) // per_sample)
        dataset_cap = round(approximate_n_voxels_dataset * self.max_dataset_covered / prod(patch))
        batch_size = max(self.UNet_min_batch_size, min(batch_size, dataset_cap))
        plan = self.assemble_plan(data_identifier, batch_size, patch, median_shape, spacing, architecture)
        plan['batch_dice'] = False  # vanilla 3d_fullres without a cascade
        plan['memory_estimate'] = {
            'method': 'uncalibrated_convolution_outputs_v1',
            'parameter_bytes_per_element': self.parameter_bytes_per_element,
            'activation_bytes_per_element': self.activation_bytes_per_element,
            'safety_fraction': self.memory_safety_fraction,
            'fixed_bytes': fixed, 'per_sample_bytes': per_sample,
            'estimated_training_bytes': fixed + batch_size * per_sample,
            'gpu_memory_target_in_gb': self.UNet_vram_target_GB,
            'checkpointing_discount': False,
        }
        return plan

    def plan_experiment(self):
        transpose_forward, transpose_backward = self.determine_transpose()
        spacing = self.determine_fullres_target_spacing()
        new_shapes = [compute_new_shape(shape, old_spacing, spacing) for old_spacing, shape in
                      zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        median_shape = np.median(new_shapes, axis=0)[transpose_forward]
        if np.any(median_shape <= 1):
            raise ValueError('MedNeXtPlanner requires a volumetric 3D dataset')
        configuration = self.get_plans_for_configuration(
            spacing[transpose_forward], median_shape, self.generate_data_identifier('3d_fullres'),
            float(prod(median_shape) * self.dataset_json['numTraining']), {})
        median_spacing = np.median(self.dataset_fingerprint['spacings'], axis=0)[transpose_forward]
        original_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], axis=0)[transpose_forward]
        plans = {
            'dataset_name': self.dataset_name, 'plans_name': self.plans_identifier,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in original_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {'3d_fullres': configuration},
            'experiment_planner_used': self.__class__.__name__, 'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint[
                'foreground_intensity_properties_per_channel'],
            # Omit initial_lr so each trainer can supply its own default. An explicit
            # global initial_lr remains supported, as in the local vanilla trainer.
            'num_epochs': 1000, 'lr_exponent': 0.9, 'save_every': 50,
        }
        copy_no_perms(join(self.raw_dataset_folder, 'dataset.json'),
                      join(nnUNet_preprocessed, self.dataset_name, 'dataset.json'))
        self.plans = plans
        self.save_plans(plans)
        return plans
