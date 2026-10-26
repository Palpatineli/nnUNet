from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform

from nnunetv2.experiment_planning.experiment_planners.residual_unets.residual_encoder_unet_planners import nnUNetPlannerResEncM
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.tversky import MemoryEfficientSoftTverskyLoss
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerTverskyLoss import (
    nnUNetTrainerTverskyLoss, nnUNetTrainerTverskyLossNoMirroring, resolve_ce_class_weights,
)
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager


@pytest.fixture(autouse=True)
def local_environment(monkeypatch, tmp_path):
    torch.set_num_threads(2)
    monkeypatch.setenv('nnUNet_compile', 'false')
    monkeypatch.setenv('nnUNet_wandb_enabled', '0')
    for key in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        path = tmp_path / key
        path.mkdir()
        monkeypatch.setenv(key, str(path))


@pytest.mark.parametrize('batch_dice', [False, True])
@pytest.mark.parametrize('do_bg', [False, True])
@pytest.mark.parametrize('shape', [(2, 3, 4, 5), (2, 3, 4, 5, 6)])
def test_half_alpha_matches_vanilla_dice_values_and_gradients(batch_dice, do_bg, shape):
    logits = torch.randn(shape, requires_grad=True)
    target = torch.randint(0, 3, (shape[0], 1, *shape[2:]))
    mask = torch.randint(0, 2, target.shape)
    options = dict(apply_nonlin=lambda x: x.softmax(1), batch_dice=batch_dice,
                   do_bg=do_bg, smooth=1e-5, ddp=False)
    dice = MemoryEfficientSoftDiceLoss(**options)(logits, target, mask)
    tversky = MemoryEfficientSoftTverskyLoss(**options, alpha=0.5)(logits, target, mask)
    torch.testing.assert_close(tversky, dice)
    torch.testing.assert_close(torch.autograd.grad(tversky, logits, retain_graph=True)[0],
                               torch.autograd.grad(dice, logits)[0])


@pytest.mark.parametrize('alpha', [0., 0.3, 0.5, 0.9, 1.])
@pytest.mark.parametrize('batch_dice', [False, True])
def test_matches_direct_multiclass_definition(alpha, batch_dice):
    logits = torch.randn(2, 4, 4, 5, requires_grad=True)
    target = torch.randint(0, 4, (2, 1, 4, 5))
    mask = torch.randint(0, 2, target.shape)
    loss = MemoryEfficientSoftTverskyLoss(lambda x: x.softmax(1), batch_dice=batch_dice,
                                         do_bg=False, smooth=1e-5, ddp=False, alpha=alpha)
    actual = loss(logits, target, mask)
    probs = logits.softmax(1)
    onehot = F.one_hot(target[:, 0], num_classes=4).movedim(-1, 1).float()
    axes = (0, 2, 3) if batch_dice else (2, 3)
    tp = (probs * onehot * mask).sum(axes)
    fp = (probs * (1 - onehot) * mask).sum(axes)
    fn = ((1 - probs) * onehot * mask).sum(axes)
    score = (tp + 5e-6) / (tp + alpha * fp + (1 - alpha) * fn + 5e-6)
    expected = -(score[1:] if batch_dice else score[:, 1:]).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


def test_default_penalizes_false_negatives_more():
    # Keep TP=1 and compare exactly one FN against exactly one FP.
    loss = MemoryEfficientSoftTverskyLoss(do_bg=False, ddp=False)
    def probability(foreground):
        p = torch.tensor(foreground).reshape(1, 1, 1, 3)
        return torch.cat((1 - p, p), dim=1)
    more_fn = loss(probability([1., 0., 0.]), torch.tensor([[[[1, 1, 0]]]]))
    more_fp = loss(probability([1., 1., 0.]), torch.tensor([[[[1, 0, 0]]]]))
    assert more_fn > more_fp
    assert loss.alpha == 0.3 and loss.beta == 0.7


@pytest.mark.parametrize('alpha', [-0.1, 1.1, float('nan'), float('inf'), '0.3', True])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match='tversky_alpha'):
        MemoryEfficientSoftTverskyLoss(alpha=alpha)


def test_missing_classes_all_ignored_and_mixed_precision():
    loss = DC_and_CE_loss({'alpha': 0.3, 'do_bg': False, 'ddp': False, 'smooth': 1e-5}, {},
                          ignore_label=3, dice_class=MemoryEfficientSoftTverskyLoss)
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    ignored = torch.full((2, 1, 4, 5), 3)
    result = loss(logits, ignored)
    result.backward()
    assert torch.isfinite(result) and torch.count_nonzero(logits.grad) == 0
    logits = torch.randn(2, 3, 4, 5, dtype=torch.bfloat16, requires_grad=True)
    absent_foreground = torch.zeros(2, 1, 4, 5, dtype=torch.long)
    result = loss(logits, absent_foreground)
    result.backward()
    assert torch.isfinite(result) and torch.isfinite(logits.grad).all()


def labels():
    return LabelManager({'background': 0, 'nerve': 1, 'muscle': 2, 'ignore': 3}, None)


def make_trainer(settings=None, regions=False, deep_supervision=False, single_head=False):
    trainer = object.__new__(nnUNetTrainerTverskyLoss)
    trainer.device = torch.device('cpu')
    trainer.is_ddp = False
    trainer.enable_deep_supervision = deep_supervision
    trainer.print_to_log_file = lambda *args, **kwargs: None
    trainer.label_manager = (LabelManager({'background': 0, 'whole': (1, 2), 'core': 2, 'ignore': 3}, [1, 2])
                             if regions else labels())
    trainer.configuration_manager = ConfigurationManager({
        'patch_size': [8, 8, 8], 'batch_dice': True, 'loss': settings or {},
        'architecture': {'arch_kwargs': {'strides': [[1, 1, 1], [2, 2, 2]] if single_head else
                                         [[1, 1, 1], [2, 2, 2], [2, 2, 2]]}}})
    return trainer


def test_class_weight_names_ids_and_lists():
    torch.testing.assert_close(resolve_ce_class_weights({'nerve': 3., 'muscle': 2.}, labels()),
                               torch.tensor([1., 3., 2.]))
    torch.testing.assert_close(resolve_ce_class_weights({'0': 0.5, '1': 3., '2': 2.}, labels()),
                               torch.tensor([0.5, 3., 2.]))
    torch.testing.assert_close(resolve_ce_class_weights([1., 4., 2.], labels()),
                               torch.tensor([1., 4., 2.]))
    assert resolve_ce_class_weights(None, labels()) is None


@pytest.mark.parametrize('weights', [[1, 2], [1, 2, 3, 4], [-1, 1, 1], [0, 1, 1],
                                    [1, float('nan'), 1], [[1], [2], [3]],
                                    {'missing': 2}, {'ignore': 2}, {'nerve': 2, '1': 3}, 'invalid'])
def test_invalid_class_weights(weights):
    with pytest.raises(ValueError):
        resolve_ce_class_weights(weights, labels())


def test_default_and_individual_ce_weights_match_pytorch():
    trainer = make_trainer()
    unweighted = trainer._build_loss()
    assert unweighted.ce.weight is None
    assert unweighted.dc.alpha == 0.3 and unweighted.dc.beta == 0.7
    trainer = make_trainer({'tversky_alpha': 0.2, 'ce_class_weights': {'nerve': 4., 'muscle': 2.}})
    loss = trainer._build_loss()
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    target = torch.randint(0, 4, (2, 1, 4, 5))
    expected_ce = F.cross_entropy(logits, target[:, 0], weight=torch.tensor([1., 4., 2.]), ignore_index=3)
    mask = target != 3
    overlap = loss.dc(logits, torch.where(mask, target, 0), loss_mask=mask)
    torch.testing.assert_close(loss(logits, target), overlap + expected_ce)
    loss(logits, target).backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('settings', [None, 'invalid', {'alpha': 0.3}, {'tversky_beta': 0.7},
                                     {'tversky_alpha': float('nan')}])
def test_invalid_plan_loss_settings(settings):
    trainer = make_trainer()
    trainer.configuration_manager.configuration['loss'] = settings
    with pytest.raises(ValueError):
        trainer._build_loss()


@pytest.mark.parametrize('single_head', [False, True])
@pytest.mark.parametrize('ddp,compiled', [(False, False), (True, False), (True, True)])
def test_deep_supervision_weights(single_head, ddp, compiled):
    trainer = make_trainer(deep_supervision=True, single_head=single_head)
    trainer.is_ddp = ddp
    trainer._do_i_compile = lambda: compiled
    loss = trainer._build_loss()
    assert isinstance(loss, DeepSupervisionWrapper)
    assert sum(loss.weight_factors) == pytest.approx(1.)
    if single_head:
        assert loss.weight_factors == (1.,)
    else:
        assert loss.weight_factors[-1] > 0 if ddp and not compiled else loss.weight_factors[-1] == 0


def test_region_weights_and_ignore_mask():
    trainer = make_trainer({'ce_class_weights': {'whole': 2., 'core': 3.}}, regions=True)
    loss = trainer._build_loss()
    logits = torch.randn(2, 2, 4, 5, 6, requires_grad=True)
    target = torch.randint(0, 2, (2, 3, 4, 5, 6))  # last channel is ignore mask
    mask = (1 - target[:, -1:]).bool()
    expected_bce = (F.binary_cross_entropy_with_logits(
        logits, target[:, :-1].float(), reduction='none', weight=torch.tensor([2., 3.]).view(1, 2, 1, 1, 1)) * mask).sum() / mask.sum()
    expected = loss.dc(logits, target[:, :-1], loss_mask=mask) + expected_bce
    torch.testing.assert_close(loss(logits, target), expected)
    loss(logits, target).backward()
    assert torch.isfinite(logits.grad).all()


def test_compiled_loss_values_and_gradients():
    loss = MemoryEfficientSoftTverskyLoss(apply_nonlin=lambda x: x.softmax(1),
                                         do_bg=False, ddp=False, alpha=0.3)
    compiled = torch.compile(loss, backend='eager', fullgraph=True)
    logits = torch.randn(2, 3, 4, 5, requires_grad=True)
    target = torch.randint(0, 3, (2, 1, 4, 5))
    a, b = loss(logits, target), compiled(logits, target)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(torch.autograd.grad(a, logits, retain_graph=True)[0],
                               torch.autograd.grad(b, logits)[0])


@pytest.fixture
def resenc_dataset(tmp_path):
    name = 'Dataset998_TverskyTest'
    raw = tmp_path / 'nnUNet_raw' / name
    preprocessed = tmp_path / 'nnUNet_preprocessed' / name
    (raw / 'imagesTr').mkdir(parents=True)
    (raw / 'labelsTr').mkdir()
    preprocessed.mkdir()
    (raw / 'imagesTr' / 'case_0000.nii.gz').touch()
    (raw / 'labelsTr' / 'case.nii.gz').touch()
    dataset_json = {'channel_names': {'0': 'MRI'}, 'labels': labels().label_dict,
                    'numTraining': 1, 'file_ending': '.nii.gz',
                    'overwrite_image_reader_writer': 'SimpleITKIO'}
    fingerprint = {'spacings': [[1., 1., 1.]], 'shapes_after_crop': [[16, 16, 16]],
                   'median_relative_size_after_cropping': 1.,
                   'foreground_intensity_properties_per_channel': {'0': {}}}
    (raw / 'dataset.json').write_text(json.dumps(dataset_json))
    (preprocessed / 'dataset_fingerprint.json').write_text(json.dumps(fingerprint))
    plans = nnUNetPlannerResEncM(name).plan_experiment()
    plans['configurations']['3d_fullres']['loss'] = {
        'tversky_alpha': 0.3, 'ce_class_weights': {'nerve': 3., 'muscle': 2.}}
    return plans, dataset_json


@pytest.mark.parametrize('trainer_class', [nnUNetTrainerTverskyLoss, nnUNetTrainerTverskyLossNoMirroring])
def test_resenc_training_resume_and_standard_inference(resenc_dataset, trainer_class):
    plans, dataset_json = resenc_dataset
    original = deepcopy(plans)
    trainer = trainer_class(plans, '3d_fullres', 0, dataset_json, device=torch.device('cpu'))
    assert plans == original
    _, _, _, mirror_axes = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    if trainer_class is nnUNetTrainerTverskyLossNoMirroring:
        assert mirror_axes is None and trainer.inference_allowed_mirroring_axes is None
    path = Path(trainer.preprocessed_dataset_folder)
    path.mkdir()
    (path / 'case.npz').touch()
    trainer.initialize()
    assert isinstance(trainer.optimizer, torch.optim.SGD)
    assert trainer.initial_lr == 1e-2 and trainer.weight_decay == 3e-5
    assert trainer.network.__class__.__name__ == 'ResidualEncoderUNet'
    patch = trainer.configuration_manager.patch_size
    data = torch.randn(2, 1, *patch)
    target = torch.randint(0, 4, (1, *patch))
    transform = DownsampleSegForDSTransform(trainer._get_deep_supervision_scales())
    targets = [t.unsqueeze(0).repeat(2, 1, 1, 1, 1)
               for t in transform(segmentation=target)['segmentation']]
    before = next(trainer.network.parameters()).detach().clone()
    assert np.isfinite(trainer.train_step({'data': data, 'target': targets})['loss'])
    assert not torch.equal(before, next(trainer.network.parameters()))
    assert np.isfinite(trainer.validation_step({'data': data, 'target': targets})['loss'])
    checkpoint = Path(trainer.output_folder) / 'checkpoint_final.pth'
    trainer.save_checkpoint(str(checkpoint))
    resumed = trainer_class(plans, '3d_fullres', 0, dataset_json, device=torch.device('cpu'))
    resumed.load_checkpoint(str(checkpoint))
    assert resumed.optimizer.state_dict()['state'] and resumed.current_epoch == 1
    resumed.network.eval()
    resumed.set_deep_supervision_enabled(False)
    with torch.no_grad():
        expected = resumed.network(data[:1])
    model_dir = Path(trainer.output_folder_base)
    (model_dir / 'plans.json').write_text(json.dumps(plans))
    (model_dir / 'dataset.json').write_text(json.dumps(dataset_json))
    predictor = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False,
                               use_mirroring=True, use_gaussian=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,))
    assert predictor.trainer_name == trainer_class.__name__
    if trainer_class is nnUNetTrainerTverskyLossNoMirroring:
        assert predictor.allowed_mirroring_axes is None
    else:
        # Disable TTA for numerical comparison with a single forward pass.
        predictor.use_mirroring = False
    predictor.network.eval()
    with torch.no_grad():
        actual = predictor.predict_sliding_window_return_logits(data[0])
    torch.testing.assert_close(actual.float(), expected[0], atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize('patch', [[8, 8], [8, 32], [8, 8, 8], [4, 32, 32]])
def test_no_mirroring_keeps_rotation_and_patch_configuration(patch):
    trainer = make_trainer()
    trainer.configuration_manager.configuration['patch_size'] = patch
    expected = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    assert trainer.inference_allowed_mirroring_axes is not None
    no_mirror = object.__new__(nnUNetTrainerTverskyLossNoMirroring)
    no_mirror.__dict__.update(trainer.__dict__)
    actual = no_mirror.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    assert actual[:2] == expected[:2]
    np.testing.assert_array_equal(actual[2], expected[2])
    assert actual[3] is None and no_mirror.inference_allowed_mirroring_axes is None
