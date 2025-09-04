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
        # Support both uint8 [0,255] and float [0,1] inputs
        if x.dtype.is_floating_point:
            return x
        return x.float().div(255.)

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


class LatentDynamicsLoss(nn.MSELoss):
    """Custom loss function for latent dynamics."""
    def __init__(self):
        super().__init__()

    def forward(
            self, 
            transition_latent, 
            encoded_latent,
            reward_pred,
            reward_target
        ):
        # Align dtypes to avoid Float/Double mismatches
        encoded_latent = encoded_latent.to(dtype=transition_latent.dtype)
        reward_target = reward_target.to(dtype=reward_pred.dtype)
        # both losses are measured by the MSE
        transition_loss = super().forward(transition_latent, encoded_latent)
        reward_loss = super().forward(reward_pred, reward_target)
        return transition_loss + reward_loss

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds) # calls self.forward()