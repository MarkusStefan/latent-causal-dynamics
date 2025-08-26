'''
Utilities & helper funcs
'''

import numpy as np
import torch 
from torch import nn


class NormalizeImg(nn.Module):
    """Simple module to normalize image pixel values."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # We assume the input image is in the range [0, 255]
        return x.div(255.)

class Flatten(nn.Module):
    """Simple module to flatten a tensor."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    

def _get_out_shape(input_shape, layers):
    """Helper to compute the output shape of a sequence of layers."""
    dummy_input = torch.zeros(1, *input_shape)
    with torch.no_grad():
        for layer in layers:
            dummy_input = layer(dummy_input)
    return dummy_input.shape[1:]