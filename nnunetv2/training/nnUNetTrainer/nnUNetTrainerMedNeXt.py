"""MedNeXt trainers; all data processing and training loops remain inherited."""
from copy import deepcopy

import torch

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerMedNeXt(nnUNetTrainer):
    """Vanilla SGD recipe with plan validation and the standard decoder interface."""
    default_initial_lr = 1e-2

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # The local vanilla trainer pops continue_training. Avoid mutating callers'
        # plans and support ordinary constructor use as well as the training CLI.
        plans = deepcopy(plans)
        plans.setdefault('continue_training', False)
        plans.setdefault('initial_lr', self.default_initial_lr)
        super().__init__(plans, configuration, fold, dataset_json, device)

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        if configuration_manager.network_arch_class_name != 'dynamic_network_architectures.architectures.mednext.MedNeXt':
            raise ValueError('Use MedNeXt plans with nnUNetTrainerMedNeXt')
        network = nnUNetTrainer.build_network_architecture(
            plans_manager, configuration_manager, num_input_channels, num_output_channels, enable_deep_supervision)
        divisibility = [1, 1, 1]
        for stride in network.encoder.strides:
            for axis, s in enumerate(stride):
                divisibility[axis] *= s
        if len(configuration_manager.patch_size) != 3 or any(
                size < 1 or size % divisor for size, divisor in zip(configuration_manager.patch_size, divisibility)):
            raise ValueError('MedNeXt patch_size must be 3D and divisible by cumulative strides; replan after geometry edits')
        return network

    def _build_loss(self):
        # A tiny volume may yield only two resolution levels / one decoder head.
        # Vanilla's final zero weight would otherwise normalize an all-zero vector.
        if self.enable_deep_supervision and len(self._get_deep_supervision_scales()) == 1:
            self.enable_deep_supervision = False
            try:
                loss = super()._build_loss()
            finally:
                self.enable_deep_supervision = True
            return DeepSupervisionWrapper(loss, [1.])
        return super()._build_loss()


class nnUNetTrainerMedNeXtAdamW(nnUNetTrainerMedNeXt):
    """MedNeXt AdamW defaults, with all other behavior inherited from SGD."""
    default_initial_lr = 1e-3

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr,
                                      weight_decay=self.weight_decay, eps=1e-4)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, self.lr_exponent)
