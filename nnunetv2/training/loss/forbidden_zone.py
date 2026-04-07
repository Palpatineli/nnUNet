import torch
from torch import nn
from typing import Callable
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class MemoryEfficientSoftDiceLossWithForbiddenPenalty(nn.Module):
    """Dice loss for single-channel predictions with a forbidden-zone penalty.

    Expects:
    - `x`: network output probabilities/logits for the positive class with shape (b, 1, ...).
    - `y`: one-hot-like tensor with shape (b, 2, ...) where channel 0 = expected/positive,
      channel 1 = forbidden. (If more channels are present, the last channel is treated as
    the forbidden mask and the remaining channels are treated as expected regions.)
    """

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, penalty_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.penalty_weight = float(penalty_weight)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor, loss_mask=None):
        # apply nonlinearity to network output if requested (probabilities)
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        # assume y is one-hot-like
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            # forbidden is last channel
            forbidden_mask = (y == 2).to(torch.float32)
            expected_mask = (y == 1).to(torch.float32)

            # if there are multiple expected channels sum them to a single target
            if expected_mask.shape[1] > 1:
                expected_mask = expected_mask.sum(dim=1, keepdim=True)

            if not self.do_bg:
                # if do_bg is False, drop background channel if present
                # assume expected_mask channel 0 is background when multiple classes
                pass

            sum_gt = expected_mask.sum(axes, dtype=torch.float32) if loss_mask is None else (expected_mask * loss_mask).sum(axes, dtype=torch.float32)

        # compute intersections and sums
        if loss_mask is None:
            intersect = (x * expected_mask).sum(axes, dtype=torch.float32)
            sum_pred = x.sum(axes, dtype=torch.float32)
        else:
            intersect = (x * expected_mask * loss_mask).sum(axes, dtype=torch.float32)
            sum_pred = (x * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0, dtype=torch.float32)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0, dtype=torch.float32)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0, dtype=torch.float32)

            intersect = intersect.sum(0, dtype=torch.float32)
            sum_pred = sum_pred.sum(0, dtype=torch.float32)
            sum_gt = sum_gt.sum(0, dtype=torch.float32)

        dc = (2 * intersect + self.smooth) / (sum_gt + sum_pred + float(self.smooth)).clamp_min(1e-8)
        dice_loss = -dc.mean()

        # forbidden penalty
        if loss_mask is None:
            intersect_forbidden = (x * forbidden_mask).sum(axes, dtype=torch.float32)
        else:
            intersect_forbidden = (x * forbidden_mask * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                intersect_forbidden = AllGatherGrad.apply(intersect_forbidden).sum(0, dtype=torch.float32)
            intersect_forbidden = intersect_forbidden.sum(0, dtype=torch.float32)

        penalty_frac = (intersect_forbidden / (sum_pred + self.eps)).mean()

        return dice_loss + self.penalty_weight * penalty_frac


class RobustCrossEntropyLossWithForbiddenPenalty(nn.Module):
    """CrossEntropy loss that expects a one-hot `target` where the last channel is forbidden.

    Behavior:
    - Builds integer CE target by taking argmax over the non-forbidden channels.
    - Computes normal CE loss on `input` (logits).
    - Adds forbidden penalty = mean( intersection(pred_prob_pos, forbidden) / (sum(pred_prob_pos)+eps) ) * penalty_weight
    """

    def __init__(self, weight=None, ignore_index: int = -100, reduction='mean', label_smoothing: float = 0,
                 penalty_weight: float = 1.0, eps: float = 1e-8, pos_class: int = 1):
        super().__init__()
        # reuse internal PyTorch CE loss for the base term
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)
        self.penalty_weight = float(penalty_weight)
        self.eps = float(eps)
        self.pos_class = int(pos_class)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target: one-hot-like (b, C, ...), last channel is forbidden
        if input.ndim != target.ndim:
            raise ValueError('RobustCrossEntropyLossWithForbiddenPenalty expects one-hot target with same ndim as input')

        # separate forbidden channel
        forbidden = (target == 2)
        classes = (target == 1)

        # build integer target by argmax over classes
        ce_target = classes.argmax(dim=1)

        # base CE loss (input are logits)
        base_loss = self.ce(input, ce_target.long())

        # predicted probability for positive class (pos_class index)
        probs = torch.softmax(input, dim=1)
        if probs.shape[1] <= self.pos_class:
            pred_prob = probs[:, -1:]
        else:
            pred_prob = probs[:, self.pos_class:self.pos_class+1]

        axes = tuple(range(2, pred_prob.ndim))
        intersect_forbidden = (pred_prob * forbidden).sum(dim=axes, dtype=torch.float32)
        sum_pred = pred_prob.sum(dim=axes, dtype=torch.float32)

        penalty_frac = (intersect_forbidden / (sum_pred + self.eps)).mean()

        return base_loss + self.penalty_weight * penalty_frac


class DC_and_CE_forbidden(nn.Module):
    """Compound loss combining the forbidden-zone Dice and CE losses.

    Expects `target` to be one-hot-like where the last channel is the forbidden mask.
    """

    def __init__(self, soft_dice_kwargs: dict, ce_kwargs: dict, weight_ce=1, weight_dice=1,
                 dice_class=MemoryEfficientSoftDiceLossWithForbiddenPenalty,
                 ce_class=RobustCrossEntropyLossWithForbiddenPenalty):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dc = dice_class(**soft_dice_kwargs)
        self.ce = ce_class(**ce_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        return self.weight_ce * ce_loss + self.weight_dice * dc_loss
