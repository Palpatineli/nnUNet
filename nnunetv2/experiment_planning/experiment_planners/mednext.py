from typing import List, Tuple
import numpy as np
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner


class MedNeXtPlanner(ExperimentPlanner):
    def get_plans_for_configuration(
        self,
        spacing: np.ndarray | Tuple[float, ...] | List[float],
        median_shape: np.ndarray | Tuple[int, ...],
        data_identifier: str,
        approximate_n_voxels_dataset: float,
        _cache: dict,
    ) -> dict:
        plan = {
            "data_identifier": data_identifier,
            "preprocessor_name": self.preprocessor_name,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "median_image_size_in_voxels": median_shape,
            "spacing": spacing,
            "normalization_schemes": normalization_schemes,
            "use_mask_for_norm": mask_is_used_for_norm,
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_seg": resampling_seg.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "resampling_fn_seg_kwargs": resampling_seg_kwargs,
            "resampling_fn_probabilities": resampling_softmax.__name__,
            "resampling_fn_probabilities_kwargs": resampling_softmax_kwargs,
            "architecture": architecture_kwargs,
        }
        return plan
