"""Integration checks for the neighboring MedNeXt implementation and nnU-Net."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform

from dynamic_network_architectures.architectures.abstract_arch import test_submodules_loadable as check_submodules
from dynamic_network_architectures.architectures.mednext import MedNeXt
from dynamic_network_architectures.building_blocks.mednext import GlobalResponseNorm, MedNeXtBlock
from nnunetv2.experiment_planning.experiment_planners.mednext import MedNeXtPlanner
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMedNeXt import (
    nnUNetTrainerMedNeXt, nnUNetTrainerMedNeXtAdamW,
)
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


@pytest.fixture(autouse=True)
def local_environment(monkeypatch, tmp_path):
    # No logging, compilation caches, or temporary artifacts outside test roots.
    monkeypatch.setenv('nnUNet_compile', 'false')
    monkeypatch.setenv('nnUNet_wandb_enabled', '0')
    for key, subdir in [('nnUNet_raw', 'raw'), ('nnUNet_preprocessed', 'preprocessed'),
                        ('nnUNet_results', 'results')]:
        directory = tmp_path / subdir
        directory.mkdir()
        monkeypatch.setenv(key, str(directory))
    torch.set_num_threads(2)


@pytest.fixture
def dataset(tmp_path):
    dataset_name = 'Dataset999_MedNeXtTest'
    raw = tmp_path / 'raw' / dataset_name
    preprocessed = tmp_path / 'preprocessed' / dataset_name
    raw.mkdir()
    preprocessed.mkdir()
    (raw / 'imagesTr').mkdir()
    (raw / 'labelsTr').mkdir()
    # Reader selection is explicit; planning uses the supplied fingerprint.
    (raw / 'imagesTr' / 'case_0000.nii.gz').touch()
    (raw / 'labelsTr' / 'case.nii.gz').touch()
    dataset_json = {'channel_names': {'0': 'MRI'}, 'labels': {'background': 0, 'organ': 1},
                    'numTraining': 1, 'file_ending': '.nii.gz',
                    'overwrite_image_reader_writer': 'SimpleITKIO'}
    fingerprint = {'spacings': [[4., 1., 1.]], 'shapes_after_crop': [[16, 32, 32]],
                   'median_relative_size_after_cropping': 1.,
                   'foreground_intensity_properties_per_channel': {'0': {}}}
    (raw / 'dataset.json').write_text(json.dumps(dataset_json))
    (preprocessed / 'dataset_fingerprint.json').write_text(json.dumps(fingerprint))
    return dataset_name, dataset_json


def architecture_kwargs(**overrides):
    kwargs = dict(input_channels=1, n_stages=3, features_per_stage=[4, 8, 16], conv_op=nn.Conv3d,
                  kernel_sizes=[[1, 3, 3], [3, 3, 3], [3, 3, 3]],
                  strides=[[1, 1, 1], [1, 2, 2], [2, 2, 2]], num_classes=2,
                  n_conv_per_stage=1, n_conv_per_stage_decoder=1, deep_supervision=True)
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize('norm_type', ['group', 'layer'])
@pytest.mark.parametrize('shape', [(8, 16, 16), (9, 17, 19)])
def test_registration_shapes_and_gradients(norm_type, shape):
    net = MedNeXt(**architecture_kwargs(norm_type=norm_type))
    check_submodules(net)
    outputs = net(torch.randn(2, 1, *shape))
    assert outputs[0].shape == (2, 2, *shape)
    assert outputs[1].shape == (2, 2, shape[0], (shape[1] + 1) // 2, (shape[2] + 1) // 2)
    sum(o.square().mean() for o in outputs).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(isinstance(b.grn, GlobalResponseNorm) for b in net.modules() if isinstance(b, MedNeXtBlock))
    for prefix in ('encoder.stem', 'encoder.stages', 'encoder.down_blocks',
                   'decoder.up_blocks', 'decoder.stages', 'decoder.seg_layers'):
        assert any(n.startswith(prefix) for n in net.state_dict())
    net.decoder.deep_supervision = False
    assert isinstance(net(torch.randn(2, 1, *shape)), torch.Tensor)


def test_feature_map_estimator_matches_executed_convolutions():
    net = MedNeXt(**architecture_kwargs())
    elements = []
    handles = [module.register_forward_hook(lambda m, args, out: elements.append(out[0].numel()))
               for module in net.modules() if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d))]
    net(torch.randn(2, 1, 8, 16, 16))
    for handle in handles:
        handle.remove()
    assert net.compute_conv_feature_map_size((8, 16, 16)) == sum(elements)


def test_checkpointing_matches_plain_gradients():
    torch.manual_seed(7)
    plain = MedNeXt(**architecture_kwargs(drop_path_rate=0.15))
    checked = MedNeXt(**architecture_kwargs(drop_path_rate=0.15, outside_block_checkpointing=True))
    checked.load_state_dict(plain.state_dict())
    x = torch.randn(2, 1, 8, 16, 16)
    torch.manual_seed(42)
    result = plain(x)
    sum(o.square().mean() for o in result).backward()
    torch.manual_seed(42)
    checked_result = checked(x)
    sum(o.square().mean() for o in checked_result).backward()
    for a, b in zip(result, checked_result):
        torch.testing.assert_close(a, b)
    for a, b in zip(plain.parameters(), checked.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


def test_grn_identity_and_large_half_precision_response():
    grn = GlobalResponseNorm(8, 3)
    x = torch.full((2, 8, 8, 8, 8), 10000., dtype=torch.float16, requires_grad=True)
    y = grn(x)
    torch.testing.assert_close(x, y)
    y.float().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in grn.parameters())
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('checkpointing', [False, True])
def test_compiled_wrapper_and_supervision_toggle(checkpointing):
    net = MedNeXt(**architecture_kwargs(outside_block_checkpointing=checkpointing))
    compiled = torch.compile(net, backend='eager', fullgraph=True)
    x = torch.randn(2, 1, 8, 16, 16)
    sum(o.square().mean() for o in compiled(x)).backward()
    trainer = object.__new__(nnUNetTrainerMedNeXt)
    trainer.network, trainer.is_ddp = compiled, False
    trainer.set_deep_supervision_enabled(False)
    assert compiled(x).shape == (2, 2, 8, 16, 16)
    assert all(p.grad is not None for p in net.parameters())


def test_profile_requires_an_accessible_accelerator(monkeypatch):
    from nnunetv2.experiment_planning.profile_mednext import profile
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='No CUDA/ROCm device'):
        profile({}, {})


@pytest.mark.parametrize('overrides', [dict(n_stages=1), dict(features_per_stage=[4, 8]),
                                      dict(n_conv_per_stage=[1]), dict(kernel_sizes=2),
                                      dict(strides=2), dict(norm_type='invalid'),
                                      dict(drop_path_rate=1.), dict(conv_op=nn.Conv2d)])
def test_invalid_architectures(overrides):
    with pytest.raises(ValueError):
        MedNeXt(**architecture_kwargs(**overrides))


def make_plans(dataset):
    name, dataset_json = dataset
    planner = MedNeXtPlanner(name, architecture_overrides={'features_per_stage': 4,
                                                          'n_conv_per_stage': 1,
                                                          'n_conv_per_stage_decoder': 1})
    return planner, planner.plan_experiment(), dataset_json


def test_native_planning_and_preprocessing(dataset, tmp_path):
    planner, plans, dataset_json = make_plans(dataset)
    assert list(plans['configurations']) == ['3d_fullres']
    assert 'initial_lr' not in plans
    saved = json.loads((tmp_path / 'preprocessed' / dataset[0] / 'MedNeXtPlans.json').read_text())
    manager = PlansManager(saved)
    config = manager.get_configuration('3d_fullres')
    assert config.preprocessor_class is DefaultPreprocessor
    assert config.previous_stage_name is None and config.next_stage_names is None
    assert not config.batch_dice
    assert config.pool_op_kernel_sizes[1][0] == 1
    assert config.network_arch_init_kwargs['kernel_sizes'][0][0] == 1
    net = get_network_from_plans(config.network_arch_class_name, config.network_arch_init_kwargs,
                                 config.network_arch_init_kwargs_req_import, 1, 2, deep_supervision=False)
    assert net(torch.randn(1, 1, *config.patch_size)).shape[2:] == tuple(config.patch_size)
    data = np.random.default_rng(0).normal(size=(1, 16, 32, 32)).astype(np.float32)
    seg = np.zeros_like(data, dtype=np.int16)
    seg[:, 4:10, 8:20, 8:20] = 1
    normalized, resampled, properties = config.preprocessor_class(verbose=False).run_case_npy(
        data, seg, {'spacing': [4., 1., 1.]}, manager, config, dataset_json)
    assert normalized.shape == resampled.shape == data.shape
    assert np.isfinite(normalized).all()
    assert set(np.unique(resampled)) == {0, 1}
    assert 'bbox_used_for_cropping' in properties
    estimate = config.configuration['memory_estimate']
    assert estimate['estimated_training_bytes'] <= 8 * 1024 ** 3 * estimate['safety_fraction']


def test_planner_budget_capacity_and_spacing(dataset):
    name, _ = dataset
    planner = MedNeXtPlanner(name)
    high = planner.get_plans_for_configuration([3., 1., 1.], [128, 256, 256], 'test', 1e9, {})
    planner.UNet_vram_target_GB = 2
    low = planner.get_plans_for_configuration([3., 1., 1.], [128, 256, 256], 'test', 1e9, {})
    assert np.prod(low['patch_size']) < np.prod(high['patch_size'])
    assert low['batch_size'] >= 2
    for plan in (high, low):
        kwargs = plan['architecture']['arch_kwargs']
        assert kwargs['features_per_stage'] == [min(32 * 2 ** i, 320) for i in range(kwargs['n_stages'])]
        assert kwargs['n_conv_per_stage'] == [2] * kwargs['n_stages']
        assert plan['memory_estimate']['estimated_training_bytes'] <= (
            plan['memory_estimate']['gpu_memory_target_in_gb'] * 1024 ** 3 * 0.8)
    planner = MedNeXtPlanner(name, overwrite_target_spacing=[2., 0.8, 0.8], kernel_size=5)
    plans = planner.plan_experiment()
    assert plans['configurations']['3d_fullres']['spacing'] == [2., 0.8, 0.8]
    assert max(max(k) for k in plans['configurations']['3d_fullres']['architecture']['arch_kwargs']['kernel_sizes']) == 5


@pytest.mark.parametrize('trainer_class,expected_lr', [(nnUNetTrainerMedNeXt, 1e-2),
                                                     (nnUNetTrainerMedNeXtAdamW, 1e-3)])
def test_training_checkpoint_and_standard_predictor(dataset, tmp_path, trainer_class, expected_lr):
    _, plans, dataset_json = make_plans(dataset)
    original = deepcopy(plans)
    trainer = trainer_class(plans, '3d_fullres', 0, dataset_json, device=torch.device('cpu'))
    assert plans == original
    assert trainer.initial_lr == expected_lr
    # Initialize the real network/optimizer/loss without requiring a preprocessed file database.
    config = trainer.configuration_manager
    Path(trainer.preprocessed_dataset_folder).mkdir()
    # infer_dataset_class recognizes this extension; no dataset loading is needed here.
    (Path(trainer.preprocessed_dataset_folder) / 'case.npz').touch()
    trainer.initialize()
    expected_optimizer = torch.optim.AdamW if trainer_class is nnUNetTrainerMedNeXtAdamW else torch.optim.SGD
    assert isinstance(trainer.optimizer, expected_optimizer)
    if expected_optimizer is torch.optim.AdamW:
        assert trainer.optimizer.param_groups[0]['eps'] == 1e-4
        assert trainer.optimizer.param_groups[0]['betas'] == (0.9, 0.999)
    shape = config.patch_size
    data = torch.randn(2, 1, *shape)
    transform = DownsampleSegForDSTransform(trainer._get_deep_supervision_scales())
    targets = transform(segmentation=torch.randint(0, 2, (1, *shape)))['segmentation']
    targets = [t.unsqueeze(0).repeat(2, 1, 1, 1, 1) for t in targets]
    before = trainer.network.encoder.stem.weight.detach().clone()
    assert np.isfinite(trainer.train_step({'data': data, 'target': targets})['loss'])
    assert not torch.equal(before, trainer.network.encoder.stem.weight)
    trainer.lr_scheduler.step(100)
    assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(expected_lr * 0.9 ** 0.9)
    trainer.set_deep_supervision_enabled(False)
    assert not trainer.network.decoder.deep_supervision
    trainer.network.eval()
    with torch.no_grad():
        expected = trainer.network(data[:1])
    trainer.set_deep_supervision_enabled(True)
    checkpoint = Path(trainer.output_folder) / 'checkpoint_final.pth'
    trainer.save_checkpoint(str(checkpoint))
    resumed = trainer_class(plans, '3d_fullres', 0, dataset_json, device=torch.device('cpu'))
    resumed.load_checkpoint(str(checkpoint))
    assert resumed.current_epoch == 1
    assert resumed.optimizer.state_dict()['state']
    resumed.set_deep_supervision_enabled(False)
    resumed.network.eval()
    with torch.no_grad():
        torch.testing.assert_close(resumed.network(data[:1]), expected)
    model_dir = Path(trainer.output_folder_base)
    (model_dir / 'plans.json').write_text(json.dumps(plans))
    (model_dir / 'dataset.json').write_text(json.dumps(dataset_json))
    predictor = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False,
                               use_mirroring=False, use_gaussian=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,))
    predictor.network.eval()
    with torch.no_grad():
        torch.testing.assert_close(predictor.network(data[:1]), expected)
        logits = predictor.predict_sliding_window_return_logits(data[0])
    assert logits.shape == (2, *shape)
    torch.testing.assert_close(logits.float(), expected[0], rtol=2e-3, atol=2e-3)


def test_explicit_learning_rate_and_bad_patch(dataset):
    _, plans, dataset_json = make_plans(dataset)
    plans['initial_lr'] = 2e-4
    trainer = nnUNetTrainerMedNeXtAdamW(plans, '3d_fullres', 0, dataset_json, device=torch.device('cpu'))
    assert trainer.initial_lr == 2e-4
    patch = list(trainer.configuration_manager.patch_size)
    patch[1] += 1
    trainer.configuration_manager.configuration['patch_size'] = patch
    with pytest.raises(ValueError, match='divisible'):
        trainer.build_network_architecture(trainer.plans_manager, trainer.configuration_manager, 1, 2)


def test_single_decoder_supervision(dataset):
    _, plans, dataset_json = make_plans(dataset)
    config = plans['configurations']['3d_fullres']
    config['patch_size'] = [8, 8, 8]
    config['architecture']['arch_kwargs'].update(
        n_stages=2, features_per_stage=[4, 8], kernel_sizes=3,
        strides=[[1, 1, 1], [2, 2, 2]], expansion_ratio=4,
        n_conv_per_stage=1, n_conv_per_stage_decoder=1)
    trainer = nnUNetTrainerMedNeXt(plans, '3d_fullres', 0, dataset_json, device=torch.device('cpu'))
    trainer.network = trainer.build_network_architecture(trainer.plans_manager, trainer.configuration_manager, 1, 2)
    loss = trainer._build_loss()
    outputs = trainer.network(torch.randn(2, 1, 8, 8, 8))
    result = loss(outputs, [torch.randint(0, 2, (2, 1, 8, 8, 8))])
    result.backward()
    assert torch.isfinite(result)
