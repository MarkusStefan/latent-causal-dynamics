"""Run simple short evaluation episodes on a DMControl task for LCDM, SAC, and LOOP.
Saves results to notebooks/quick_dmc_results.csv and a PNG.

This script expects the repository layout and the project's DMCEnv wrapper to be available.
Make sure `dm_control` is installed and available in the Python environment.
"""
import os
import sys
import argparse
import torch
import pandas as pd

# Ensure repo root is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

try:
    from env import DMCEnv
    from lcdm import LCDM
    from sac import SACAgent
    from loop import LOOPAgent, LOOPPlanner
except Exception:
    print("Failed to import project modules. Ensure you're running inside the repo and dependencies are installed.")
    raise


def run_short_eval(domain='cartpole', task='swingup', img_size=64, episodes=3, steps_per_ep=100, device='cpu'):
    # create env
    env = DMCEnv(domain_name=domain, task_name=task, from_pixels=True, img_size=img_size)
    obs = env.reset()

    # derive action dim
    action_spec = env.get_actions()
    try:
        action_shape = action_spec.shape
    except Exception:
        try:
            action_shape = action_spec.tensor_spec.shape
        except Exception:
            action_shape = (1,)
    action_dim = int(action_shape[0]) if len(action_shape) > 0 else 1

    # instantiate agents
    lcdm = LCDM(obs_shape=tuple(obs.shape), action_dim=action_dim, img_size=img_size, latent_dim=16, device=device)
    sac = SACAgent(obs_shape=tuple(obs.shape), action_dim=action_dim, device=device, use_encoder=True, encoder_params={'in_channels':3,'num_channels':32,'img_size':img_size,'latent_dim':16})
    loop_agent = LOOPAgent(latent_dim=16, action_dim=action_dim, device=device)
    loop_planner = LOOPPlanner(latent_dim=16, action_dim=action_dim, horizon=5, num_samples=128, sigma=0.3, device=device)

    results = {'lcdm':[], 'sac':[], 'loop':[]}

    for alg in ['lcdm','sac','loop']:
        for ep in range(episodes):
            obs = env.reset()
            ep_return = 0.0
            for t in range(steps_per_ep):
                if alg == 'lcdm':
                    a = lcdm.act(obs)
                elif alg == 'sac':
                    a = sac.act(obs)
                else:
                    with torch.no_grad():
                        z = lcdm.encode(obs.unsqueeze(0))
                    a = loop_planner.plan(lcdm.transition_model, lcdm.reward_model, loop_agent.value_net, z)
                next_obs, reward, done, _ = env.step(a)
                ep_return += float(reward)
                obs = next_obs
            results[alg].append(ep_return)
            print(f"{alg} ep {ep+1} return={ep_return:.3f}")

    # save results
    os.makedirs('notebooks', exist_ok=True)
    df = pd.DataFrame({k:pd.Series(v) for k,v in results.items()})
    csv_path = 'notebooks/quick_dmc_results.csv'
    df.to_csv(csv_path, index=False)
    ax = df.plot(kind='box')
    ax.set_title(f'Quick DMC benchmark: {domain}/{task}')
    fig = ax.get_figure()
    png_path = 'notebooks/quick_dmc_returns.png'
    fig.savefig(png_path)
    print(f"Wrote {csv_path} and {png_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='cartpole')
    parser.add_argument('--task', default='swingup')
    parser.add_argument('--img-size', default=64, type=int)
    parser.add_argument('--episodes', default=3, type=int)
    parser.add_argument('--steps', default=100, type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_short_eval(domain=args.domain, task=args.task, img_size=args.img_size, episodes=args.episodes, steps_per_ep=args.steps, device=args.device)
