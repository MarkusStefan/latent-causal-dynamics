"""
Soft Actor-Critic (SAC) implementation compatible with this repo's env wrapper.

This implementation optionally uses the same `Encoder` from `models.py` when
working with pixel observations. It provides a simple training step and an
`act` method for interaction.
"""
from typing import Optional, Tuple
import math
import torch
from torch import nn, optim
import torch.nn.functional as F

from models import Encoder


class ObsReplayBuffer:
    """Simple replay buffer storing raw observations (tensors).
    Stores tensors on CPU to avoid GPU memory growth during long runs.
    """
    def __init__(self, capacity: int = 100_000, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.storage = []

    def add(self, obs, action, reward, next_obs, done):
        item = (obs.cpu(), action.cpu(), torch.tensor(reward).cpu(), next_obs.cpu(), torch.tensor(done).cpu())
        if len(self.storage) >= self.capacity:
            self.storage.pop(0)
        self.storage.append(item)

    def __len__(self):
        return len(self.storage)

    def sample(self, batch_size: int = 256, device: str = "cpu"):
        assert len(self.storage) >= batch_size
        idx = torch.randint(0, len(self.storage), (batch_size,))
        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for i in idx:
            o, a, r, no, d = self.storage[int(i)]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            dones.append(d)
        return dict(
            obs=torch.stack(obs, 0).to(device),
            action=torch.stack(actions, 0).to(device),
            reward=torch.stack(rewards, 0).to(device).float(),
            next_obs=torch.stack(next_obs, 0).to(device),
            done=torch.stack(dones, 0).to(device).float(),
        )


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x.log1p() - (-x).log1p())


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        mu = self.mean(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + eps * std
        action = torch.tanh(pre_tanh)
        # log prob correction for tanh
        log_prob = -0.5 * ((pre_tanh - mu) / (std + 1e-8)).pow(2) - log_std - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob.sum(dim=-1)
        # correction
        log_prob -= (1 - action.pow(2) + 1e-6).log().sum(dim=-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class SACAgent:
    def __init__(
        self,
        obs_shape: tuple,
        action_dim: int,
        device: str = "cpu",
        use_encoder: bool = False,
        encoder_params: dict | None = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: Optional[float] = None,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim

        # encoder for pixels -> latent
        self.use_encoder = use_encoder
        if use_encoder:
            assert encoder_params is not None
            self.encoder = Encoder(**encoder_params).to(device)
            obs_dim = encoder_params["latent_dim"]
        else:
            # assume flat observation vector
            obs_dim = int(torch.tensor(obs_shape).prod().item())

        # actor and critics
        self.actor = GaussianActor(obs_dim, action_dim).to(device)
        self.q1 = Critic(obs_dim, action_dim).to(device)
        self.q2 = Critic(obs_dim, action_dim).to(device)
        self.q1_target = Critic(obs_dim, action_dim).to(device)
        self.q2_target = Critic(obs_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # optimizers
        params = list(self.actor.parameters())
        if use_encoder:
            params += list(self.encoder.parameters())
        self.actor_opt = optim.Adam(params, lr=lr)
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        # entropy temperature
        if alpha is None:
            self.target_entropy = -action_dim
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = None
            self.alpha = alpha

        self.replay = ObsReplayBuffer(device=device)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs may be CHW image or flat vector; ensure batch dim
        if self.use_encoder:
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)
            z = self.encoder(obs.to(self.device))
            return z
        else:
            x = obs.float().to(self.device)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return x.view(x.shape[0], -1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        z = self.encode_obs(obs)
        if deterministic:
            mu, _ = self.actor.forward(z)
            return mu.squeeze(0).cpu()
        a, _ = self.actor.sample(z)
        return a.squeeze(0).cpu()

    def update(self, batch_size: int = 256):
        if len(self.replay) < batch_size:
            return None
        batch = self.replay.sample(batch_size, device=self.device)
        obs = self.encode_obs(batch["obs"])  # [B, obs_dim]
        next_obs = self.encode_obs(batch["next_obs"])  # [B, obs_dim]
        action = batch["action"].to(self.device).float()
        reward = batch["reward"].to(self.device).float()
        done = batch["done"].to(self.device).float()

        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs)
            q1n = self.q1_target(next_obs, next_action)
            q2n = self.q2_target(next_obs, next_action)
            qn = torch.min(q1n, q2n)
            alpha = (self.log_alpha.exp() if self.log_alpha is not None else self.alpha)
            target = reward + (1 - done) * self.gamma * (qn - alpha * next_logp)

        # critic loss
        q1_pred = self.q1(obs, action)
        q2_pred = self.q2(obs, action)
        loss_q = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        self.q_opt.zero_grad()
        loss_q.backward()
        self.q_opt.step()

        # actor loss
        a_pi, logp_pi = self.actor.sample(obs)
        q1_pi = self.q1(obs, a_pi)
        q2_pi = self.q2(obs, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = (self.log_alpha.exp() if self.log_alpha is not None else self.alpha)
        loss_pi = (alpha * logp_pi - q_pi).mean()
        self.actor_opt.zero_grad()
        loss_pi.backward()
        self.actor_opt.step()

        # temperature update
        if self.log_alpha is not None:
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()

        # soft update targets
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.mul_(1 - self.tau)
            tp.data.add_(self.tau * p.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.mul_(1 - self.tau)
            tp.data.add_(self.tau * p.data)

        return dict(loss_q=float(loss_q.item()), loss_pi=float(loss_pi.item()))

