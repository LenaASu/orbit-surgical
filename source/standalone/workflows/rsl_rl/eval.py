"""
Script to run PPO policies to pick and lift the suture needle and save the policies with top 5 highest success rate.


.. code-block:: bash

    ${IsaacLab_PATH}/isaaclab.sh -p source/standalone/workflows/rsl_rl/eval.py

    ~/IsaacLab/isaaclab.sh -p source/standalone/workflows/rsl_rl/eval.py
  
"""

"""Launch Omniverse Toolkit first."""

import argparse
from pathlib import Path
import pandas as pd
from isaaclab.app import AppLauncher
# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using PPO for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes to simulate.")
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
    default="logs/rsl_rl/needle_lift",
)

FILE_PATH = Path(__file__).resolve().parent
save_path = FILE_PATH / "results" / "ppo_top5.csv"

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab.app  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import orbit.surgical.tasks  # noqa: F401

def eval_checkpoint_ppo(env, agent_cfg, checkpoint_path):
    """Play with RSL-RL agent and print results summary."""
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(str(checkpoint_path))
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, info = env.reset()
    
    episode_id = 1
    episode_step = 0
    
    step_cnt = 0
    success_log = 0
    timeout_log = 0
    drop_log = 0
    success_cnt = 0
    timeout_cnt = 0
    drop_cnt = 0

    while simulation_app.is_running() and episode_id <= args_cli.num_episodes:
        
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            actions = policy(obs)
        obs, rewards, dones, info = env.step(actions)

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

            # reset
            episode_step = 0
            episode_id += 1
            obs, info = env.reset()
    return {
        "checkpoint": checkpoint_path.name,
        "path": str(checkpoint_path),
        "episodes": episode_id - 1,
        "success": success_cnt,
        "timeout": timeout_cnt,
        "drop": drop_cnt,
        "success_rate": success_cnt / (episode_id - 1) * 100.0
    }

def main():
    checkpoint_dir = Path(args_cli.checkpoint_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    checkpoint_files = sorted(checkpoint_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]),)
    checkpoint_files = [p for p in checkpoint_files if int(p.stem.split("_")[1]) >= 200]
    results = []

    for checkpoint_path in checkpoint_files:
        print(f"\n[INFO] Evaluating {checkpoint_path.name}")
        result = eval_checkpoint_ppo(env=env, agent_cfg=agent_cfg, checkpoint_path=checkpoint_path)
        results.append(result)

        print(f"{result['checkpoint']}:")
        print(f"success {result['success']}/{result['episodes']}")
        print(f"{result['success_rate']:.1f}%")
        print(f"timeout {result['timeout']}")
        print(f"drop {result['drop']}")
    
    results = sorted(results, key=lambda x:x["success_rate"], reverse=True)

    # print summary
    print("\n" + "=" * 70)
    print("Top 5 PPO Checkpoints")
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

    df = pd.DataFrame(results[:5])
    df.to_csv(save_path, index=False)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()