# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run and eval BC models in robomimic."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play and eval policy trained using robomimic for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Isaac-Lift-Needle-PSM-IK-Abs-v0", help="Name of the task.")
# parser.add_argument("--checkpoint", type=str, default="source/standalone/environments/data/datasets/lift_n_dataset_Abs_100.hdf5", help="Hdf5 checkpoint to load.")
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="logs/robomimic/Isaac-Lift-Needle-PSM-IK-Abs-v0/bc/20260704/models",
    help="Directory contains model_epoch_*.pth checkpoints."
)
parser.add_argument("--num_episodes", type=str, default=50, help="Episodes playing for evaluation.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import pandas as pd
import gymnasium as gym

import robomimic  # noqa: F401
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import isaaclab.app  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401

FILE_PATH = Path(__file__).resolve().parent
save_path = FILE_PATH / "results" / "bc_top5.csv"

def convert_obs(obs_dict):
    """Convert Isaac Lab observations to a single-env robomimic NumPy dict."""
    policy_obs = obs_dict["policy"]

    obs_keys = {
        "joint_pos",
        "joint_vel",
        "object_position",
        "target_object_position",
    }

    return {
        key: policy_obs[key][0].detach().cpu().numpy() for key in obs_keys
    }

def eval_checkpoint(env, checkpoint_path):
    """Play with robomimic agent and print results summary."""
    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=device, verbose=False)
    
    obs_dict, _ = env.reset()
    episode_id = 1
    episode_step = 0
    
    step_cnt = 0
    success_log = 0
    timeout_log = 0
    drop_log = 0
    success_cnt = 0
    timeout_cnt = 0
    drop_cnt = 0

    num_episodes = args_cli.num_episodes 
  
    success_steps = []
    episode_reward = 0.0
    total_rewards = []
    episode_lengths = []

    while simulation_app.is_running() and episode_id <= num_episodes:
        with torch.inference_mode():
            obs = convert_obs(obs_dict)
            
            action_np = policy(ob=obs)
            action = torch.as_tensor(action_np, device=env.unwrapped.device, dtype=torch.float32).reshape(1, -1)
        obs_dict, reward, terminated, truncated, info = env.step(action)
        # print("info", info)
        # print("env.unwrapped.termination_manager:", env.unwrapped.termination_manager)
        # print("env.unwrapped.scene.keys():", env.unwrapped.scene.keys())
        
        step_cnt += 1
        episode_step += 1
        episode_reward += reward[0].item()
        
        if step_cnt % 100 == 0:
            print(f"Running: step={step_cnt}, episode={episode_id}")
            
        if terminated | truncated:
            success_log = info["log"]["Episode_Termination/object_lifted"]
            timeout_log = info["log"]["Episode_Termination/time_out"]
            drop_log = info["log"]["Episode_Termination/object_dropping"]

            if success_log == 1:
                success_cnt += 1
                success_steps.append(episode_step)
                
            if timeout_log == 1:
                timeout_cnt += 1

            if drop_log == 1:
                drop_cnt += 1
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_step)
            
            # reset
            episode_step = 0
            episode_reward = 0.0
            # print(info)
            # print("success:", success_cnt)
            # print("timeout: ", timeout_cnt)
            episode_id += 1
            obs_dict, _ = env.reset()

    return {
        "checkpoint": checkpoint_path.name,
        "path": str(checkpoint_path),
        "episodes": episode_id - 1,
        "success": success_cnt,
        "timeout": timeout_cnt,
        "drop": drop_cnt,
        "mean_success_steps": (float(np.mean(success_steps) if success_steps else 0)),
        "mean_rewards": (float(np.mean(total_rewards) if total_rewards else 0.0)),
        "mean_episode_length": (float(np.mean(episode_lengths) if episode_lengths else 0)),
        "success_rate": success_cnt / (episode_id - 1) * 100.0 
    }


def main():
    checkpoint_dir = Path(args_cli.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {checkpoint_dir}"
        )
    
    checkpoint_files = sorted(Path(args_cli.checkpoint_dir).glob("model_epoch_*.pth"), key=lambda p: int(p.stem.split("_")[2]))
    # checkpoint_files = [p for p in checkpoint_files if int(p.stem.split("_")[1]) >= 200]
    

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    results = []
    

    for checkpoint_path in checkpoint_files:
        print(f"\n[INFO] Evaluating {checkpoint_path.name}")
        result = eval_checkpoint(env=env, checkpoint_path=checkpoint_path)
        results.append(result)

        print(f"{result['checkpoint']}:")
        print(f"success {result['success']}/{result['episodes']}")
        print(f"success_rate {result['success_rate']:.1f}%")
        print(f"timeout {result['timeout']}/{result['episodes']}")
        print(f"drop {result['drop']}/{result['episodes']}")

    results = sorted(results, key=lambda x:x["success_rate"], reverse=True)

    print("\n" + "=" * 70)
    print("Top 5 BC Checkpoints")
    print("=" * 70)
    for i, r in enumerate(results[:5], start=1):
        print(
            f"{i}. {r['checkpoint']:<15} "
            f"success_rate={r['success_rate']:.1f}% "
            f"success={r['success']}/{r['episodes']} "
            f"timeout={r['timeout']} "
            f"drop={r['drop']}"
            f"mean_reward={r['mean_rewards']}"
            f"mean_success_step={r['mean_success_steps']}"
            f"mean_episode_length={r['mean_episode_length']}"
        )
    print("=" * 70)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results[:5])
    df.to_csv(save_path, index=False)
    print(f"[INFO] Saved results to {save_path}.")
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()