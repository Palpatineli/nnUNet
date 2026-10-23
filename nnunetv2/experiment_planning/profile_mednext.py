"""Synthetic MedNeXt training memory measurements on CUDA or ROCm.

Run with python -m nnunetv2.experiment_planning.profile_mednext --help.
This does not train on dataset images or measure segmentation quality.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def profile(plans, dataset_json, configuration='3d_fullres', optimizer_name='adamw',
            batch_size=None, steps=3, warmup=1, checkpointing=None, compile_network=False):
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA/ROCm device is accessible; accelerator memory cannot be measured')
    if steps < 1 or warmup < 1:
        raise ValueError('At least one warmup step and one measurement step are required')
    manager = PlansManager(plans)
    config = manager.get_configuration(configuration)
    labels = manager.get_label_manager(dataset_json)
    batch_size = config.batch_size if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    kwargs = deepcopy(config.network_arch_init_kwargs)
    if config.network_arch_class_name != 'dynamic_network_architectures.architectures.mednext.MedNeXt':
        raise ValueError('Memory profiling requires MedNeXt plans')
    if checkpointing is not None:
        kwargs['outside_block_checkpointing'] = checkpointing
    channels = len(dataset_json.get('channel_names', dataset_json.get('modality', {})))
    net = get_network_from_plans(config.network_arch_class_name, kwargs,
                                 config.network_arch_init_kwargs_req_import, channels,
                                 labels.num_segmentation_heads, deep_supervision=True).cuda().train()
    if compile_network:
        net = torch.compile(net)
    lr = plans.get('initial_lr', 1e-3 if optimizer_name == 'adamw' else 1e-2)
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=3e-5, eps=1e-4)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    else:
        raise ValueError('optimizer_name must be adamw or sgd')
    dice_kwargs = {'batch_dice': config.batch_dice, 'smooth': 1e-5, 'ddp': False}
    if labels.has_regions:
        base_loss = DC_and_BCE_loss({}, {**dice_kwargs, 'do_bg': True},
                                    use_ignore_label=labels.has_ignore_label,
                                    dice_class=MemoryEfficientSoftDiceLoss)
    else:
        base_loss = DC_and_CE_loss({**dice_kwargs, 'do_bg': False}, {}, weight_ce=1, weight_dice=1,
                                   ignore_label=labels.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
    if compile_network:
        base_loss.dc = torch.compile(base_loss.dc)
    scales = (1 / np.cumprod(np.asarray(kwargs['strides']), axis=0))[:-1]
    weights = np.asarray([1 / 2 ** i for i in range(len(scales))], dtype=float)
    if len(weights) > 1:
        weights[-1] = 0
    loss_function = DeepSupervisionWrapper(base_loss, weights / weights.sum())
    shape = config.patch_size
    data = torch.randn(batch_size, channels, *shape, device='cuda')
    target_channels = labels.num_segmentation_heads if labels.has_regions else 1
    target = torch.randint(0, 2 if labels.has_regions else labels.num_segmentation_heads,
                           (batch_size, target_channels, *shape), device='cuda')
    if labels.has_regions and labels.has_ignore_label:
        target = torch.cat((target, torch.zeros_like(target[:, :1])), dim=1)
    targets = [F.interpolate(target.float(), size=[round(s * d) for s, d in zip(scale, shape)],
                              mode='nearest-exact').long() for scale in scales]
    try:
        scaler = torch.amp.GradScaler('cuda')
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler()

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = loss_function(net(data), targets)
        if not torch.isfinite(loss):
            raise RuntimeError('Synthetic training loss is not finite')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    for _ in range(warmup):
        step()
    if not optimizer.state:
        raise RuntimeError('Warmup did not allocate optimizer state; increase --warmup to allow GradScaler to settle')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    losses = [step() for _ in range(steps)]
    torch.cuda.synchronize()
    return {
        'backend': 'rocm' if torch.version.hip else 'cuda', 'torch_version': torch.__version__,
        'hip_version': torch.version.hip, 'cuda_version': torch.version.cuda,
        'device': torch.cuda.get_device_name(), 'optimizer': optimizer_name,
        'batch_size': batch_size, 'patch_size': shape, 'precision': 'autocast_float16',
        'checkpointing': kwargs.get('outside_block_checkpointing', False),
        'compiled': compile_network, 'steps': steps, 'warmup': warmup,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'losses': losses,
        'planner_memory_estimate': config.configuration.get('memory_estimate'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', type=Path, required=True)
    parser.add_argument('--dataset-json', type=Path, help='Defaults to dataset.json beside the plans')
    parser.add_argument('--configuration', default='3d_fullres')
    parser.add_argument('--optimizer', choices=('sgd', 'adamw'), default='adamw')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--checkpointing', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--compile', action='store_true', dest='compile_network')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plans = json.loads(args.plans.read_text())
    dataset_json = json.loads((args.dataset_json or args.plans.parent / 'dataset.json').read_text())
    try:
        result = profile(plans, dataset_json, args.configuration, args.optimizer, args.batch_size,
                         args.steps, args.warmup, args.checkpointing, args.compile_network)
    except RuntimeError as error:
        parser.exit(1, f'{error}\n')
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + '\n')
    print(output)


if __name__ == '__main__':
    main()
