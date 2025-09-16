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
import torch
import torch.nn as nn
from torch.nn import functional as F

class LatentTransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        
        # We need a total input dimension for the linear layers
        input_dim = latent_dim + action_dim
        
        # The first linear layer is where we'll apply the mask
        self.fc1 = nn.Linear(input_dim, 256)
        
        # Subsequent layers are standard
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, latent_dim)
        
        # Create the causal mask
        self._create_causal_mask(latent_dim, action_dim)

    def _create_causal_mask(self, latent_dim, action_dim):
        # A simple causal mask where each latent variable predicts itself
        # and the action affects all latent variables.
        self.register_buffer("mask", torch.zeros(latent_dim, latent_dim + action_dim))
        
        # The latent variables can influence themselves.
        for i in range(latent_dim):
            self.mask[i, i] = 1.0
            
        # The action can influence all latent variables.
        self.mask[:, latent_dim:] = 1.0
    
    def forward(self, z, a):
        # Concatenate the latent state and action
        x = torch.cat([z, a], dim=-1)
        
        # Apply the mask to the weights of the first layer
        # This is where the causal regularization happens
        masked_weights = self.fc1.weight * self.mask
        
        # Perform the first linear transformation with the masked weights
        x = F.linear(x, masked_weights, self.fc1.bias)
        
        # Pass through the rest of the network
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x





class RewardModel():
    pass





class PCMI():
    pass
