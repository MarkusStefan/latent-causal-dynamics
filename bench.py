"""
Simple benchmark runner to compare LCDM and SAC on DMControl tasks.

Usage examples:
    python bench.py --alg lcdm --domain cartpole --task swingup --episodes 5
    python bench.py --alg sac --domain cartpole --task swingup --episodes 5

This script writes per-episode returns to `<alg>_<domain>_<task>_results.csv`.
"""
import argparse
import csv
import random
import torch

from env import DMCEnv

from lcdm import LCDM
from sac import SACAgent


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def run_lcdm(domain, task, episodes, steps_per_ep, device, img_size, latent_dim):
    env = DMCEnv(domain, task, from_pixels=True, pixels_only=True, img_size=img_size)
    obs = env.reset()
    action_dim = env.action_space.shape[-1]
    lcdm = LCDM(obs_shape=tuple(obs.shape), action_dim=action_dim, img_size=img_size, latent_dim=latent_dim, device=device)

    rows = []
    total_steps = 0
    for ep in range(episodes):
        obs = env.reset()
        ep_return = 0.0
        for t in range(steps_per_ep):
            a = lcdm.act(obs)
            next_obs, reward, done, _ = env.step(a)
            with torch.no_grad():
                z_t = lcdm.encode(obs.unsqueeze(0)).squeeze(0)
                z_t_1 = lcdm.encode(next_obs.unsqueeze(0)).squeeze(0)
            lcdm.buffer.add_transition(z_t, a, reward.view(1), z_t_1)
            loss = lcdm.optimize_step(batch_size=128)
            obs = next_obs
            ep_return += float(reward.item())
            total_steps += 1
            if total_steps % 500 == 0:
                lcdm.update_pcmci(batch_size=1024)
        rows.append((ep + 1, ep_return, loss if loss is not None else 0.0))
        print(f"LCDM Episode {ep+1}/{episodes} return={ep_return:.2f} loss={loss}")
    return rows


def run_sac(domain, task, episodes, steps_per_ep, device, img_size):
    env = DMCEnv(domain, task, from_pixels=True, pixels_only=True, img_size=img_size)
    obs = env.reset()
    action_dim = env.action_space.shape[-1]

    encoder_params = dict(in_channels=obs.shape[0], num_channels=32, img_size=img_size, latent_dim=32)
    sac = SACAgent(obs_shape=obs.shape, action_dim=action_dim, device=device, use_encoder=True, encoder_params=encoder_params)

    rows = []
    for ep in range(episodes):
        obs = env.reset()
        ep_return = 0.0
        for t in range(steps_per_ep):
            a = sac.act(obs)
            next_obs, reward, done, _ = env.step(a)
            sac.replay.add(obs, a, reward, next_obs, done)
            stats = sac.update(batch_size=128)
            obs = next_obs
            ep_return += float(reward.item())
        rows.append((ep + 1, ep_return, stats if stats is not None else {}))
        print(f"SAC Episode {ep+1}/{episodes} return={ep_return:.2f} stats={stats}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", choices=["lcdm", "sac"], default="lcdm")
    parser.add_argument("--domain", type=str, default="cartpole")
    parser.add_argument("--task", type=str, default="swingup")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps_per_ep", type=int, default=250)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.alg == "lcdm":
        rows = run_lcdm(args.domain, args.task, args.episodes, args.steps_per_ep, args.device, args.img_size, args.latent_dim)
    else:
        rows = run_sac(args.domain, args.task, args.episodes, args.steps_per_ep, args.device, args.img_size)

    out_name = f"{args.alg}_{args.domain}_{args.task}_results.csv"
    with open(out_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "extra"])
        for r in rows:
            writer.writerow(list(r))
    print(f"Wrote results to {out_name}")


if __name__ == "__main__":
    main()
