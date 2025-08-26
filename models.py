'''
Neural Models
'''
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F

from utils import NormalizeImg, Flatten, _get_out_shape



class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels, img_size, latent_dim):
        super().__init__()
        
        # Define the convolutional part of the encoder
        conv_layers = nn.Sequential(
            NormalizeImg(),
            nn.Conv2d(in_channels, num_channels, kernel_size=7, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2), nn.ReLU()
        )
        
        # Calculate the output size of the convolutional layers
        # This is needed to know the input size for the linear layer
        conv_out_shape = _get_out_shape((in_channels, img_size, img_size), conv_layers)
        
        # Combine all the layers into a single network
        self.encoder = nn.Sequential(
            conv_layers,
            Flatten(),
            # nn.Linear(torch.prod(conv_out_shape), latent_dim)
            nn.Linear(np.prod(conv_out_shape), latent_dim)
        )

    def forward(self, x):
        # The forward pass is as simple as calling the sequential container
        return self.encoder(x)



class CausalGraph():
    pass



class LatentTransitionModel():
    pass





class RewardModel():
    pass





class PCMI():
    pass
