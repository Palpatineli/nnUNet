"""Memory-efficient soft Tversky with the same interface as nnU-Net's Dice loss."""
from math import isfinite
from numbers import Real

import torch
from torch import nn

from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class MemoryEfficientSoftTverskyLoss(nn.Module):
    """Return negative mean Tversky; alpha weights FP and beta=1-alpha weights FN.

    T = (TP + smooth/2) / (TP + alpha*FP + beta*FN + smooth/2).
    Half-smoothing preserves nnU-Net's Dice smoothing convention at alpha=0.5.
    The negative score follows nnU-Net's Dice convention (1-T differs by a constant).
    """
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 ddp=True, alpha=0.3):
        super().__init__()
        if (isinstance(alpha, bool) or not isinstance(alpha, Real) or
                not isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError('tversky_alpha must be a finite number in [0, 1]')
        if not isinstance(smooth, Real) or not isfinite(smooth) or smooth < 0:
            raise ValueError('smooth must be finite and nonnegative')
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = float(smooth)
        self.ddp = ddp
        self.alpha = float(alpha)
        self.beta = 1. - self.alpha

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.reshape(y.shape[0], 1, *y.shape[1:])
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = (y_onehot if loss_mask is None else y_onehot * loss_mask).sum(axes, dtype=torch.float32)

        if not self.do_bg:
            x = x[:, 1:]
        # As in memory-efficient Dice, only spatial reductions accumulate in FP32.
        # No full-sized FP/FN/TN tensors need to be retained for backward.
        intersect = (x * y_onehot if loss_mask is None else x * y_onehot * loss_mask).sum(
            axes, dtype=torch.float32)
        sum_pred = (x if loss_mask is None else x * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)
            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        # TP + alpha*FP + beta*FN == alpha*sum_pred + beta*sum_gt
        denominator = self.alpha * sum_pred + self.beta * sum_gt + self.smooth / 2
        score = (intersect + self.smooth / 2) / denominator.clamp_min(1e-8)
        return -score.mean()
