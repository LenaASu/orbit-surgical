"""
Script to run TD3 policies to pick and lift the suture needle and save the policies with top 5 highest success rate.


.. code-block:: bash

    ${IsaacLab_PATH}/isaaclab.sh -p source/standalone/workflows/skrl/td3_eval.py

    ~/IsaacLab/isaaclab.sh -p source/standalone/workflows/skrl/td3_eval.py
  
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
    default="Isaac-Lift-Needle-PSM-IK-Rel-v0", 
    help="Name of the task."
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="logs/skrl/lift/2026-08-22_15-36-33_td3_torch/best",
)
parser.add_argument("--algorithm", type=str, default="TD3")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])

FILE_PATH = Path(__file__).resolve().parent
save_path = FILE_PATH / "results" / "eval" / "td3_top5.csv"

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

import copy
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab.app  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

# from isaaclab.assets import RigidObject
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
# from isaaclab.utils.math import subtract_frame_transforms

import orbit.surgical.tasks  # noqa: F401

def eval_checkpoint_td3(env, agent_cfg, checkpoint_path):
    # set memory_size to 1 to reduce replay buffer
    eval_cfg = copy.deepcopy(agent_cfg)
    eval_cfg["memory"]["memory_size"] = 1

    runner = Runner(env, eval_cfg)
    runner.agent.load(str(checkpoint_path))

    action_dim = env.action_space.shape[-1]

    # Patch TD3 action bounds
    runner.agent._min_actions = torch.full((action_dim,), -1.0, device=env.device)
    runner.agent._max_actions = torch.full((action_dim,), 1.0, device=env.device)
    
    obs, info = env.reset()
    print("env action_manager", env.action_manager)

    rows = [] # records of each timestep
    results = [] # records of each checkpoint

    episode_id = 1
    episode_step = 0
    step_cnt = 0

    success_cnt = 0
    timeout_cnt = 0
    drop_cnt = 0
    num_episodes = 50

    while simulation_app.is_running() and episode_id <= num_episodes:
        with torch.inference_mode():
            actions = runner.agent.act(obs, states=None, timestep=100000, timesteps=100000)
            action = actions[0]
            
        obs, rewards, terminated, truncated, info = env.step(action)
        dones = terminated | truncated

        # observations
        # robot: RigidObject = env.scene["robot"]
        # -- end-effector frame
        ee_frame_sensor = env.scene["ee_frame"]
        tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.scene.env_origins
        # tcp_rest_position_b, _ = subtract_frame_transforms(
        #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], tcp_rest_position
        # )
        # tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
        # -- object frame
        object_data: RigidObjectData = env.scene["object"].data
        object_position = object_data.root_pos_w - env.scene.env_origins
        # object_position_b, _ = subtract_frame_transforms(
        #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_position
        # )
        # object_orientation = object_data.root_quat_w
        # # -- target object frame
        # # desired_pose = env.unwrapped.command_manager.get_command("object_pose")
        # desired_pose = env.command_manager.get_command("object_pose")

        action_cpu = action[0].detach().cpu().numpy()
        # print("actions:", actions)
        # print("action:", action)
        # print("action_cpu:", action_cpu)
        ee_cpu = tcp_rest_position[0].detach().cpu().numpy()
        object_cpu = object_position[0].detach().cpu().numpy()
        reward_cpu = rewards[0].detach().cpu().item()

        # print(type(actions))
        # print(len(actions))
        # print(action.shape)
        # print(action_cpu.shape)

        rows.append({
            "checkpoint": checkpoint_path.name,
            "episode": episode_id,
            "step": episode_step,

            # 8D action: x,y,z,qw,qx,qy,qz,gripper
            # 7D action: x,y,z,drx,dry,drz,gripper
            "action_0": float(action_cpu[0]),
            "action_1": float(action_cpu[1]),
            "action_2": float(action_cpu[2]),
            "action_3": float(action_cpu[3]),
            "action_4": float(action_cpu[4]),
            "action_5": float(action_cpu[5]),
            # "action_6": float(action_cpu[6]),
            "gripper": float(action_cpu[6]),

            "ee_x": float(ee_cpu[0]),
            "ee_y": float(ee_cpu[1]),
            "ee_z": float(ee_cpu[2]),

            "object_x": float(object_cpu[0]),
            "object_y": float(object_cpu[1]),
            "object_z": float(object_cpu[2]),

            "ee_object_distance": float(np.linalg.norm(ee_cpu - object_cpu)),
            "reward": reward_cpu,
        })

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

    # rows.append({
    #         "checkpoint": checkpoint_path.name,
    #         "episode": episode_id,
    #         "step": episode_step,
    
    #         "action_0": action_cpu[0],
    #         "action_1": action_cpu[1],
    #         "action_2": action_cpu[2],
    #         "action_3": action_cpu[3],
    #         "action_4": action_cpu[4],
    #         "action_5": action_cpu[5],
    #         "action_6": action_cpu[6],
    #         "gripper": action_cpu[7],
    
    #         "ee_x": ee_cpu[0],
    #         "ee_y": ee_cpu[1],
    #         "ee_z": ee_cpu[2],
    
    #         "object_x": object_cpu[0],
    #         "object_y": object_cpu[1],
    #         "object_z": object_cpu[2],
    
    #         "reward": rewards,
    #     })

    results.append({
        "checkpoint": checkpoint_path.name,
        "path": str(checkpoint_path),
        "success": success_cnt,
        "timeout": timeout_cnt,
        "drop": drop_cnt,
        "success_rate": success_cnt / episodes * 100.0 if episodes > 0 else 0.0,
    })
    
    # return {
    #     "checkpoint": checkpoint_path.name,
    #     "path": str(checkpoint_path),
    #     "episodes": episodes,
    #     "8D action": actions,
    #     "ee_pos": tcp_rest_position.detach().cpu(),
    #     "object_pos": object_position.detach().cpu(),
    #     "success": success_cnt,
    #     "timeout": timeout_cnt,
    #     "drop": drop_cnt,
    #     "reward": rewards,
    #     "success_rate": success_cnt / episodes * 100.0 if episodes > 0 else 0.0,
    # }
    return rows, results

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
    
    rows = []
    results = []

    for checkpoint_path in checkpoint_files:
        print(f"\n[INFO] Evaluating {checkpoint_path.name}")
        ckpt_rows, ckpt_results = eval_checkpoint_td3(env=env, agent_cfg=agent_cfg, checkpoint_path=checkpoint_path)
        rows.extend(ckpt_rows)
        results.extend(ckpt_results)
        # print(results)

        # print(f"{result['checkpoint']}:")
        # print(f"success {result['success']}/{result['episodes']}")
        # print(f"{result['success_rate']:.1f}%")
        # print(f"timeout {result['timeout']}")
        # print(f"drop {result['drop']}")
    
    results = sorted(results, key=lambda x:x["success_rate"], reverse=True)


    print("\n" + "=" * 70)
    print("Top 5 TD3 Checkpoints")
    print("=" * 70)
    for i, r in enumerate(results[:5], start=1):
        print(
            f"{i}. {r['checkpoint']:<15} "
            f"success_rate={r['success_rate']:.1f}% "
            f"success={r['success']} "
            f"timeout={r['timeout']} "
            f"drop={r['drop']}"
        )
    print("=" * 70)

    # save files
    df = pd.DataFrame(results[:5])
    df.to_csv(save_path, index=False)

    trajectory_save_path = (
        FILE_PATH / "results" / "eval" / "td3_trajectories.csv"
        )

    trajectory_df = pd.DataFrame(rows)
    trajectory_df.to_csv(trajectory_save_path, index=False)

    print(f"Saved trajectories to: {trajectory_save_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()