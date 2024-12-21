from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class PreTrainer(nnUNetTrainer):
    def run_training(self):
        res = super().run_training() 
        # TODO: compare with original
