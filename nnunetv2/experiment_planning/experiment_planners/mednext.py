from typing import List, Tuple
import numpy as np
from dynamic_network_architectures.architectures.mednext import MedNeXt
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import (
    ExperimentPlanner,
)
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_patch_size


class MedNeXtPlanner(ExperimentPlanner):
    def __init__(
        self,
        dataset_name_or_id: str | int,
        gpu_memory_target_in_gb: float = 8,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "medNeXt",
        overwrite_target_spacing: List[float] | Tuple[float, ...] | None = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name,
            overwrite_target_spacing,
            suppress_transpose,
        )
        self.UNet_class = MedNeXt

    def get_plans_for_configuration(
        self,
        spacing: np.ndarray | Tuple[float, ...] | List[float],
        median_shape: np.ndarray | Tuple[int, ...],
        data_identifier: str,
        approximate_n_voxels_dataset: float,
        _cache: dict,
    ) -> dict:
        plan = super().get_plans_for_configuration(spacing, median_shape, data_identifier,
                                                   approximate_n_voxels_dataset, _cache)
        spacing = np.array(spacing)
        tmp = 1 / spacing
        if len(spacing) == 3:
            initial_patch_size = [int(round(i)) for i in tmp * (256 ** 3 / np.prod(tmp)) ** (1 / 3)]
        elif len(spacing) == 2:
            initial_patch_size = [int(round(i)) for i in tmp * (2048 ** 2 / np.prod(tmp)) ** (1 / 2)]
        else:
            raise RuntimeError()

        patch_size, _ = get_patch_size(spacing, initial_patch_size,
                                       self.UNet_featuremap_min_edge_length, 999999)
        architecture_kwargs = {
            "network_class_name": self.UNet_class.__module__ + '.' + self.UNet_class.__name__,            
            'arch_kwargs': {
                "n_stages": 4,
                "features_per_stage": 32,
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "kernel_sizes": [3, 3],
                "expansion_ratio": 4,
                "strides": [2, 2, 2, 2],
                "deep_supervision": True,
                "n_conv_per_stage": [2, 2, 2, 2, 2, 2, 2, 2, 2],
                "norm_type": "group",
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'nonlin'),
        }
        plan['patch_size'] = patch_size
        plan['architecture'] = architecture_kwargs
        return plan
