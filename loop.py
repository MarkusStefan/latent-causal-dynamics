"""LOOP-style H-step lookahead baseline.

This is a pragmatic implementation: sample K action sequences, roll forward
H steps in the latent model, use a learned terminal value net to evaluate
the final latent state, then pick the first action of the best sequence.
"""
from typing import Optional
import torch
from torch import nn, optim


class TerminalValue(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class LOOPPlanner:
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        horizon: int = 5,
        num_samples: int = 256,
        sigma: float = 0.3,
        device: str = "cpu",
    ):
        self.H = horizon
        self.K = num_samples
        self.sigma = sigma
        self.action_dim = action_dim
        self.device = device
        self.mean = torch.zeros(self.H, action_dim, device=device)

    @torch.no_grad()
    def plan(self, transition_model, reward_model, value_model, z_t: torch.Tensor, graph=None) -> torch.Tensor:
        # z_t: [1, latent_dim]
        z0 = z_t.repeat(self.K, 1)
        # sample actions
        noise = self.sigma * torch.randn(self.K, self.H, self.action_dim, device=self.device)
        actions = (self.mean.unsqueeze(0) + noise).clamp(-1.0, 1.0)

        z = z0
        returns = torch.zeros(self.K, device=self.device)
        for t in range(self.H):
            a = actions[:, t, :]
            # call transition model with provided causal graph if present
            z = transition_model(z, a, graph) if graph is not None else transition_model(z, a, getattr(transition_model, '__call__', lambda : None)())
            r = reward_model(z, a)
            returns += (0.99 ** t) * r

        # terminal value
        v = value_model(z)
        returns = returns + (0.99 ** self.H) * v
        best = returns.argmax()
        return actions[best, 0, :].clamp(-1.0, 1.0)


class LOOPAgent:
    def __init__(self, latent_dim: int, action_dim: int, device: str = "cpu", lr: float = 1e-3):
        self.device = device
        self.value_net = TerminalValue(latent_dim).to(device)
        self.opt = optim.Adam(self.value_net.parameters(), lr=lr)
        self.replay = []

    def update_value(self, batch_z, batch_target):
        z = batch_z.to(self.device)
        target = batch_target.to(self.device)
        pred = self.value_net(z)
        loss = nn.functional.mse_loss(pred, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def store(self, z, target):
        self.replay.append((z.cpu(), target.cpu()))

    def sample(self, batch_size=64):
        import random
        idx = random.sample(range(len(self.replay)), batch_size)
        zs = torch.stack([self.replay[i][0] for i in idx], 0)
        tg = torch.stack([self.replay[i][1] for i in idx], 0)
        return zs, tg
