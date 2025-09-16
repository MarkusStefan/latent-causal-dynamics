'''
Model Predictive Path Integral control algorithm
'''

from typing import Tuple
import torch

class MPPIPlanner():
    def __init__(
        self,
        action_dim: int,
        horizon: int = 10,
        num_samples: int = 1000,
        sigma: float = 0.5,
        lam: float = 1.0,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str = "cpu",
        gamma: float = 0.99,
    ):
        self.action_dim = action_dim
        self.H = horizon
        self.K = num_samples
        self.sigma = sigma
        self.lam = lam
        self.low = action_low
        self.high = action_high
        self.device = device
        self.gamma = gamma
        self.mean = torch.zeros(self.H, self.action_dim, device=self.device)

    @torch.no_grad()
    def plan(self, transition_model, reward_model, z_t: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        # z_t: [B, latent_dim] – assume B=1 in online setting
        assert z_t.shape[0] == 1
        # Sample action sequences
        noise = self.sigma * torch.randn(self.K, self.H, self.action_dim, device=self.device)
        actions = (self.mean.unsqueeze(0) + noise).clamp_(self.low, self.high)  # [K,H,A]
        # Rollout
        z0 = z_t.repeat(self.K, 1)  # [K, Z]
        returns = torch.zeros(self.K, device=self.device)
        z = z0
        for t in range(self.H):
            a_t = actions[:, t, :]  # [K,A]
            z = transition_model(z, a_t, graph)  # [K,Z]
            r = reward_model(z, a_t)  # [K]
            returns += (self.gamma ** t) * r
        # Importance weights
        costs = -returns
        beta = costs.min()
        w = torch.exp(-(costs - beta) / max(self.lam, 1e-6))  # [K]
        w = w / (w.sum() + 1e-8)
        # Update mean
        self.mean = (w.view(-1, 1, 1) * actions).sum(dim=0)
        return self.mean[0].clamp_(self.low, self.high)  # first action