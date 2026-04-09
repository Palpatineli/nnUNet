import numpy as np
import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.forbidden_zone import DC_and_CE_forbidden
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerForbiddenLoss(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': False,
            'smooth': 1e-5,
            'ddp': self.is_ddp,
            'apply_nonlin': softmax_helper_dim1,
        }

        ce_kwargs = {
            'weight': None,
            'ignore_index': self.label_manager.ignore_label if self.label_manager.has_ignore_label else -100,
            'penalty_weight': 1.0,
            'eps': 1e-8,
            'pos_class': 1,
        }

        loss = DC_and_CE_forbidden(soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """add rotations to compensate for the lack of mirroring"""
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        
        if dim == 2 and max(patch_size) / min(patch_size) > 1.5:
            rotation_for_DA = {
                'x': [(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                      (165 / 180 * np.pi, 195 / 180 * np.pi),
                      (255 / 180 * np.pi, 285 / 180 * np.pi),
                      (345 / 180 * np.pi, 375 / 180 * np.pi)],
                'y': (0, 0),
                'z': (0, 0)
            }
        elif dim == 3 and not do_dummy_2d_data_aug:
            ranges = [
                (-30 / 180 * np.pi, 30 / 180 * np.pi),
                (60 / 180 * np.pi, 120 / 180 * np.pi),
                (150 / 180 * np.pi, 210 / 180 * np.pi),
                (330 / 180 * np.pi, 390 / 180 * np.pi)
            ]
            rotation_for_DA = {'x': ranges, 'y': ranges, 'z': ranges}
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

