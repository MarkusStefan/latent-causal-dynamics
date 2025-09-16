'''Runner Script'''

import argparse
import torch
from lcdm import LCDM
from env import DMCEnv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="cartpole")
    parser.add_argument("--task", type=str, default="swingup")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps_per_ep", type=int, default=250)
    parser.add_argument("--device", type=str, default=device)
    args = parser.parse_args()

    # Env with pixels for encoder
    env = DMCEnv(args.domain, args.task, from_pixels=True, pixels_only=True, img_size=args.img_size)
    obs = env.reset()
    assert obs.ndim == 3, "Expect CHW image tensor from env when from_pixels=True"

    # Action dim/bounds
    action_dim = env.action_space.shape[-1]
    lcdm = LCDM(
        obs_shape=tuple(obs.shape),
        action_dim=action_dim,
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        device=args.device,
    )

    # Training loop
    total_steps = 0
    for ep in range(args.episodes):
        obs = env.reset()
        ep_return = 0.0
        for t in range(args.steps_per_ep):
            a = lcdm.act(obs)  # [A]
            next_obs, reward, done, _ = env.step(a)
            # Encode z, z'
            with torch.no_grad():
                z_t = lcdm.encode(obs.unsqueeze(0)).squeeze(0)
                z_t_1 = lcdm.encode(next_obs.unsqueeze(0)).squeeze(0)
            lcdm.buffer.add_transition(z_t, a, reward.view(1), z_t_1)
            # optimize
            loss = lcdm.optimize_step(batch_size=128)
            ep_return += float(reward.item())
            obs = next_obs
            total_steps += 1
            # periodically refine graph
            if total_steps % 500 == 0:
                lcdm.update_pcmci(batch_size=1024)
        if ep == 0:
            # skip loss in first episode as buffer is too small for actual loss
            print(f"Episode {ep+1}/{args.episodes} | Return: {ep_return:.2f}")
        else:
            print(f"Episode {ep+1}/{args.episodes} | Return: {ep_return:.2f} | Loss: {loss:.4f}")

if __name__ == "__main__":
    main()