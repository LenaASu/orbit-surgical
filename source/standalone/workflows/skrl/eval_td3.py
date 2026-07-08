"""
Script to run TD3 policies to pick and lift the suture needle and save the policies with top 5 highest success rate.


.. code-block:: bash

    ${IsaacLab_PATH}/isaaclab.sh -p source/standalone/workflows/skrl/eval_td3.py

    ~/IsaacLab/isaaclab.sh -p source/standalone/workflows/skrl/eval_td3.py
  
"""

"""Launch Omniverse Toolkit first."""

import argparse
from pathlib import Path
import pandas as pd
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using PPO for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--task", 
    type=str, 
    default="Isaac-Lift-Needle-PSM-IK-Abs-v0", 
    help="Name of the task."
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    # default="/workspace_data/orbit-surgical/logs/rsl_rl/needle_lift/test",
    default="/home/lena/Documents/GitHub/orbit-surgical/logs/skrl/lift/2026-07-06_17-45-59_td3_torch/checkpoints",
)
parser.add_argument("--algorithm", type=str, default="TD3")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])

FILE_PATH = Path(__file__).resolve().parent
save_path = FILE_PATH / "results" / "td3_top5.csv"

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# if args_cli.checkpoint is None:
#     # args_cli.checkpoint = "/workspace_data/orbit-surgical/logs/rsl_rl/needle_lift/test/model_1000.pt"
#     args_cli.checkpoint = "/home/lena/Documents/GitHub/orbit-surgical/logs/rsl_rl/needle_lift/2026-07-05_23-40-21/model_999.pt"
#     # args_cli.checkpoint = "/home/lena/Documents/GitHub/orbit-surgical/logs/rsl_rl/needle_lift/test/model_1000.pt"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import os
import torch
import skrl
import numpy as np
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab.app  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


import orbit.surgical.tasks  # noqa: F401

def eval_checkpoint_td3(env, agent_cfg, checkpoint_path):
    runner = Runner(env, agent_cfg)

    # Patch TD3 action bounds
    runner.agent._min_actions = torch.full((8,), -1.0, device=env.device)
    runner.agent._max_actions = torch.full((8,), 1.0, device=env.device)

    runner.agent.load(str(checkpoint_path))

    obs, info = env.reset()

    episode_id = 1
    episode_step = 0
    step_cnt = 0

    success_cnt = 0
    timeout_cnt = 0
    drop_cnt = 0
    num_episodes = 50

    while simulation_app.is_running() and episode_id <= num_episodes:
        with torch.inference_mode():
            actions = runner.agent.act(obs, states=None, timestep=0, timesteps=0)
            action = actions[0]
            
        obs, rewards, terminated, truncated, info = env.step(action)
        dones = terminated | truncated

        episode_step += 1
        step_cnt += 1

        if step_cnt % 100 == 0:
            print(f"Running: step={step_cnt}, episode={episode_id}")

        if dones.any():
            success_log = info["log"]["Episode_Termination/object_lifted"]
            timeout_log = info["log"]["Episode_Termination/time_out"]
            drop_log = info["log"]["Episode_Termination/object_dropping"]

            if success_log == 1:
                success_cnt += 1
            if timeout_log == 1:
                timeout_cnt += 1
            if drop_log == 1:
                drop_cnt += 1

            episode_id += 1
            episode_step = 0
            obs, info = env.reset()

    episodes = episode_id - 1
    return {
        "checkpoint": checkpoint_path.name,
        "path": str(checkpoint_path),
        "episodes": episodes,
        "success": success_cnt,
        "timeout": timeout_cnt,
        "drop": drop_cnt,
        "success_rate": success_cnt / episodes * 100.0 if episodes > 0 else 0.0,
    }

def main():
    checkpoint_dir = Path(args_cli.checkpoint_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_td3_cfg_entry_point")
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    checkpoint_files = sorted(
    checkpoint_dir.glob("agent_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    results = []

    for checkpoint_path in checkpoint_files:
        print(f"\n[INFO] Evaluating {checkpoint_path.name}")
        result = eval_checkpoint_td3(env=env, agent_cfg=agent_cfg, checkpoint_path=checkpoint_path)
        results.append(result)

        print(f"{result['checkpoint']}:")
        print(f"success {result['success']}/{result['episodes']}")
        print(f"{result['success_rate']:.1f}%")
        print(f"timeout {result['timeout']}")
        print(f"drop {result['drop']}")
    
    results = sorted(results, key=lambda x:x["success_rate"], reverse=True)


    print("\n" + "=" * 70)
    print("Top 5 TD3 Checkpoints")
    print("=" * 70)
    for i, r in enumerate(results[:5], start=1):
        print(
            f"{i}. {r['checkpoint']:<15} "
            f"success_rate={r['success_rate']:.1f}% "
            f"success={r['success']}/{r['episodes']} "
            f"timeout={r['timeout']} "
            f"drop={r['drop']}"
        )
    print("=" * 70)
    rows = []

    df = pd.DataFrame(results[:5])
    df.to_csv(save_path, index=False)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()