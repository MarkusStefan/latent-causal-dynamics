'''
Latent Causal Dynamics Model
'''

from typing import Optional
import torch
from torch import nn, optim

from models import Encoder, CausalGraph, LatentTransitionModel, RewardModel, PCMCI
from planner import MPPIPlanner
from buffer import TrajectoryReplayBuffer
from utils import LatentDynamicsLoss

# Algorithm 1. LCDM Training
# Require: Latent Transition model fθ, Encoder gθ
# 1: Initialize latent causal graph G as a complete directed graph
# 2: while θ not converged do
# 3:    // Policy learning from planning
# 4:    for step t=0...T do
# 5:        Select action given by at = Planner(fθ, gtθ(st)).
# 6:        Execute at in the environment and observe reward rt and new state st+1.
# 7:        Store the transition(st, at, rt, st+1) in Trajectory Buffer Bτ .
# 8:    end for
# 9:    // Latent transition model learning
# 10:   Update fθ(G) and gθ via Eq. (4) with Bτ (L(θ, φ) = sum_t=1^T[||fθ(zt, at, G), gθ(st+1)||2 + ||rφ(zt, at), rt||2])
# 11:   // Estimate Latent Causal graph
# 12:   latent causal graph G ← PCMCI(Bτ ).
# 13: end while


# class LCDM():

#     def __init__(self):
#         self.encoder = Encoder(...)
#         self.transition_model = LatentTransitionModel(...)
#         self.reward_model = RewardModel(...)
#         self.causal_graph = CausalGraph(...)
#         self.planner = MPPIPlanner(...)
#         self.buffer = TrajectoryReplayBuffer(...)
#         self.environment = ...  # Define your environment here
#         self.loss_func = LatentDynamicsLoss()


#     def train(self):
#         converged = False 

#         while not converged:
#             # Policy learning from planning
#             for t in range(T):
#                 latent_state = self.encoder.encode(state)
#                 action = self.planner.plan(self.transition_model, latent_state)
#                 next_observation, reward = self.environment.step(action)
#                 self.buffer.add_transition(latent_state, action, reward, next_observation)
            
#             # update models using the buffer
#             transition_latent = self.transition_model(latent_state, action, self.causal_graph)
#             encoded_latent = latent_state = self.encoder.encode(next_state)
#             reward_pred = self.reward_model(latent_state, action)
#             reward_target = reward
#             loss = self.loss_func(transition_latent, encoded_latent, reward_pred, reward_target)


class LCDM():

    def __init__(
        self,
        obs_shape: tuple,
        action_dim: int,
        img_size: int = 64,
        latent_dim: int = 16,
        num_channels: int = 32,
        device: str = "cpu",
        gamma: float = 0.99,
        lr: float = 1e-3,
        planner_horizon: int = 10,
        planner_samples: int = 512,
        pcmci_threshold: float = 0.1,
    ):
        C, H, W = obs_shape
        assert H == img_size and W == img_size, "Set env img_size to match encoder."
        self.device = device
        self.gamma = gamma

        # init models
        self.encoder = Encoder(in_channels=C, num_channels=num_channels, img_size=img_size, latent_dim=latent_dim).to(device)
        self.transition_model = LatentTransitionModel(latent_dim=latent_dim, action_dim=action_dim).to(device)
        self.reward_model = RewardModel(latent_dim=latent_dim, action_dim=action_dim).to(device)

        # causal graph & PCMCI estimator for causal discovery
        self.causal_graph = CausalGraph(latent_state_dim=latent_dim, latent_action_dim=action_dim).to(device)
        # self.pcmci = PCMCI(threshold=pcmci_threshold)
        self.pcmci = PCMCI()

        # Planner bounds assumed [-1,1]
        self.planner = MPPIPlanner(
            action_dim=action_dim,
            horizon=planner_horizon,
            num_samples=planner_samples,
            sigma=0.5,
            lam=1.0,
            action_low=-1.0,
            action_high=1.0,
            device=device,
            gamma=gamma,
        )

        # Buffer, loss, optimizers
        self.buffer = TrajectoryReplayBuffer(capacity=100_000, device=device)
        self.loss_func = LatentDynamicsLoss()
        self.opt = optim.Adam(
            list(self.encoder.parameters()) + list(self.transition_model.parameters()) + list(self.reward_model.parameters()),
            lr=lr,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs.to(self.device))

    def update_pcmci(self, batch_size: int = 2048):
        if len(self.buffer) < batch_size:
            return
        batch = self.buffer.sample(batch_size)
        adj = self.pcmci.estimate(batch["z_t"], batch["a_t"], batch["z_tp1"])
        self.causal_graph.update(adj)

    def optimize_step(self, batch_size: int = 128):
        if len(self.buffer) < batch_size:
            return -1
        batch = self.buffer.sample(batch_size)
        # Ensure float32 on the correct device
        z_t = batch["z_t"].to(self.device).float()
        a_t = batch["a_t"].to(self.device).float()
        r_t = batch["r_t"].to(self.device).float()
        z_tp1 = batch["z_tp1"].to(self.device).float()

        z_tp1_pred = self.transition_model(z_t, a_t, self.causal_graph())
        r_pred = self.reward_model(z_t, a_t)
        loss = self.loss_func(z_tp1_pred, z_tp1, r_pred, r_t)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.transition_model.parameters()) +
            list(self.reward_model.parameters()), 10.0
        )
        self.opt.step()
        return float(loss.item())

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.encode(obs.unsqueeze(0))  # [1,Z]
            a = self.planner.plan(self.transition_model, self.reward_model, z, self.causal_graph())
        return a  # [A]