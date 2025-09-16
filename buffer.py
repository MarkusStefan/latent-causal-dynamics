''' 
Replay Buffer
'''

from collections import deque
from typing import Dict
import torch
torch.set_default_dtype(torch.float32)

class TrajectoryReplayBuffer:
    
    def __init__(self, capacity: int = 100_000, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.capacity = capacity
        self.device = device
        self.data = deque(maxlen=capacity)

    def add_transition(self, z_t: torch.Tensor, a_t: torch.Tensor, r_t: torch.Tensor, z_t_1: torch.Tensor):
        self.data.append((
            z_t.detach().to(self.device).float(),
            a_t.detach().to(self.device).float(),
            r_t.detach().to(self.device).float(),
            z_t_1.detach().to(self.device).float(),
        ))

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, len(self.data), (batch_size,))
        z_t, a_t, r_t, z_t_1 = [], [], [], []
        for i in idx:
            zt, at, rt, zt1 = self.data[i]
            z_t.append(zt)
            a_t.append(at)
            r_t.append(rt)
            z_t_1.append(zt1)
        return dict(
            z_t=torch.stack(z_t, 0).to(self.device),
            a_t=torch.stack(a_t, 0).to(self.device),
            r_t=torch.stack(r_t, 0).to(self.device).squeeze(-1),
            z_t_1=torch.stack(z_t_1, 0).to(self.device),
        )