import warnings
import torch
from torch._dynamo import OptimizedModule
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def upkern(model_layer: Tensor, pretrained_layer: Tensor) -> Tensor:
    inc1, outc1, *spatial_dims1 = model_layer.shape
    inc2, outc2, *spatial_dims2 = pretrained_layer.shape
    print(inc1, outc1, spatial_dims1, inc2, outc2, spatial_dims2)

    # Please use equal in_channels in all layers for resizing pretrainer
    # Please use equal out_channels in all layers for resizing pretrainer
    assert inc1 == inc2 and outc1 == outc2, (
        f"The shape of model layer is not the same. Pretrained model: "
        f"{pretrained_layer.shape}; your network: {model_layer.shape}."
    )

    if spatial_dims1 == spatial_dims2:
        model_layer = pretrained_layer
        print(f"Key {k} loaded.")
    else:
        model_layer = torch.nn.functional.interpolate(
            pretrained_layer, size=spatial_dims1, mode="trilinear"
        )
        print(
            f"Key {k} interpolated trilinearly from {spatial_dims2}->{spatial_dims1} and loaded."
        )
    return model_layer


def load_pretrained_weights(network: nn.Module, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()), weights_only=False)
    else:
        saved_model = torch.load(fname, weights_only=False)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        ".seg_layers.",
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, (
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be "
                f"compatible with your network."
            )
            if model_dict[key].shape != pretrained_dict[key].shape:
                try:
                    if key.rsplit('.', maxsplit=1)[1] in ("bias", "norm", "dummy"):  # bias, norm and dummy layers
                        print(f"Key {key} loaded unchanged.")
                        model_dict[key] = pretrained_dict[key]
                    else:  # Conv / linear layers
                        model_dict[key] = upkern(model_dict[key], pretrained_dict[key])
                except AssertionError as e:
                    e.add_note(f"Incompatibility between model and pretrained checkpoint at key {key}")
                    raise e

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict.keys()
        and all([i not in k for i in skip_strings_in_pretrained])
    }

    model_dict.update(pretrained_dict)

    print(
        "################### Loading pretrained weights from file ",
        fname,
        "###################",
    )
    if verbose:
        print(
            "Below is the list of overlapping blocks in pretrained model and nnUNet architecture:"
        )
        for key, value in pretrained_dict.items():
            print(key, "shape", value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)
