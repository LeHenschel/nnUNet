from typing import Type
import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim


def get_matching_unpooling(conv_op: Type[_ConvNd] = None, dimension: int = None,
                           idx_unpool: bool = False) -> Type[torch.nn.Module]:
    """
    EITHER conv_op OR dimension MUST be set. Do not set both!
    Args:
        conv_op: convolution-operation --> used to infer the dimension (if it is not defined directly)
        dimension: dimension (1, 2, or 3), can not be defined together with conv_op
        idx_unpool: bool, perform index unpooling instead of conv-transpose

    Returns:
        module function to use for unpooling (with correct dimension)
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        if idx_unpool:
            return nn.MaxUnpool1d
        else:
            return nn.ConvTranspose1d
    elif dimension == 2:
        if idx_unpool:
            return nn.MaxUnpool2d
        else:
            return nn.ConvTranspose2d
    elif dimension == 3:
        if idx_unpool:
            return nn.MaxUnpool3d
        else:
            return nn.ConvTranspose3d
