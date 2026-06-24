# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run a trained Behavior Cloning policy to pick and lift the suture needle.


.. code-block:: bash

    ${IsaacLab_PATH}/isaaclab.sh -p source/standalone/environments/imitation_learning/eval_bc.py

    ~/IsaacLab/isaaclab.sh -p source/standalone/environments/imitation_learning/eval_bc.py 
  
"""

"""Launch Omniverse Toolkit first."""

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using imitation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", 
    type=str, 
    default="Isaac-Lift-Needle-PSM-IK-Abs-Play-v0", 
    help="Name of the task."
)
parser.add_argument(
    "--checkpoint", type=str,
    default="source/standalone/environments/imitation_learning/policies/bc_lift_n_100_policy_200.pt", 
    help="Pytorch model checkpoint to load."
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from isaaclab_tasks.utils import parse_env_cfg
import orbit.surgical.tasks  # noqa: F401
from train_bc_mse import BCPolicy


def main():
    """Run a trained policy from imitation/SB3 with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    # we want to have the terms in the observations returned as a concatenated tensor
    # rather than a dictionary
    env_cfg.observations.policy.concatenate_terms = True

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    IL_DIR = Path(__file__).resolve().parent
    checkpoint_path = args_cli.checkpoint
    # traj_path = IL_DIR / "policies" / "bc_lift_n_1_policy_200.pt"

    # load data
    checkpoint = torch.load(checkpoint_path, map_location=args_cli.device, weights_only=True)
    # traj = torch.load(traj_path, map_location=args_cli.device, weights_only=True)


    # acquire device
    # device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    # policy = ActorCriticPolicy.load(checkpoint, device=args_cli.device)
    policy = BCPolicy(obs_dim=checkpoint["obs_dim"], act_dim=checkpoint["act_dim"]).to(args_cli.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    # print("=" * 50)
    # print("Demo Obs -> BC Action Check")
    # print("=" * 50)

    # for idx in [0, 20, 50, 100]:
    #     demo_obs = traj[idx]["obs"].to(args_cli.device)
    #     demo_act = traj[idx]["action"].to(args_cli.device)

    #     with torch.inference_mode():
    #         pred_act = policy._predict(
    #             demo_obs,
    #             deterministic=True
    #         )

    #     print(f"\nStep {idx}")
    #     print("Demo:", demo_act.cpu().numpy().round(4))
    #     print("Pred:", pred_act.cpu().numpy().round(4))

    #     err = torch.abs(pred_act - demo_act).mean().item()
    #     print("Mean Abs Error:", err)


    # reset environment
    obs_dict, info = env.reset()
    # only cares about policy observations
    obs = obs_dict["policy"].to(args_cli.device)


    episode_id = 1
    episode_step = 0
    
    step_cnt = 0
    success_cnt = 0
    timeout_cnt = 0
    num_episodes = 50 # set number of episodes
    success_cnt = 0
    success_steps = []
    episode_reward = 0.0
    total_rewards = []
    episode_lengths = []
    
    # with torch.inference_mode():
    #     actions = policy(obs)
    #     actions = torch.clamp(actions, -1.0, 1.0)
    # # with torch.inference_mode():
    #     for i, data in enumerate(checkpoint):
    #         if not simulation_app.is_running():
    #             break

    # #         actions = data["action"].to(env.device)
    # #         if actions.ndim == 1:
    # #             actions = actions.unsqueeze(0)

    #         obs, rewards, terminated, truncated, info = env.step(actions)
    #         episode_reward += rewards.mean().item()
            
    #         success = info["log"]["Episode_Termination/object_lifted"]
    #         timeout = info["log"]["Episode_Termination/time_out"]

    #         print(
    #             "traj index:", i,
    #             "saved step:", data["step"],
    #             "action shape:", actions.shape,
    #             "reward:", rewards,
    #             "success:", success,
    #             "timeout:", timeout,
    #             "terminated:", terminated,
    #             "truncated:", truncated,
    #         )

    #         print("traj length:", len(traj))
    #         print("first saved step:", traj[0]["step"])
    #         print("last saved step:", traj[-1]["step"])
    #         print("last object_lifted in saved traj:", traj[-1]["object_lifted_log"])

    #         if (terminated | truncated).any():
    #             print("Episode ended at replay step:", i)
    #             break

    # print("Replay total reward:", episode_reward)
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            xyz = policy(obs)
            xyz = torch.clamp(xyz, -1.0, 1.0)
            actions = torch.zeros((obs.shape[0], 8), device=obs.device)
            actions[:, :3] = xyz
            actions[:, 3] = 1.0
            actions[:,4:7] = 0.0
            if episode_step < 180:
                actions[:, 7] = 1.0
            elif episode_step < 300:
                actions[:, 7] = -1.0
            else:
                actions[:, 7] = -1.0
                # lift
                # actions[:, 0] = 0.0435
                # actions[:, 1] = 0.0482
                actions[:, 2] = -0.1000
                # alpha = min((episode_step - 80) / 50.0, 1.0)
                # actions[:, 2] = (1 - alpha) * actions[:, 2] + alpha * (-0.08)
            # actions = policy(obs)
            # actions = torch.clamp(actions, min=-1, max=1)
            # for i, data in enumerate(traj):
            #     actions = data["action"].to(env.device)
            #     if actions.ndim == 1:
            #         actions = actions.unsqueeze(0)
                # actions = actions.to(env.device)

            # apply actions
            # obs_dict = env.step(actions)[0]
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            # only cares about policy observations
            obs = obs_dict["policy"].to(args_cli.device)

            # ee_pos = env.unwrapped.scene["robot"].data.body_pos_w[:, ee_body_id]
            object_pos = env.unwrapped.scene["object"].data.root_pos_w
    

            episode_reward += rewards.mean().item()
            episode_step += 1

            success_log = info["log"]["Episode_Termination/object_lifted"]
            timeout_log = info["log"]["Episode_Termination/time_out"]

            if success_log > 0 and terminated.any():
                success_cnt += 1
                success_steps.append(episode_step)
                episode_lengths.append(episode_step)
            
            if timeout_log > 0 and truncated.any():
                timeout_cnt += 1

            # print("obs", obs.shape, obs[0, :10])
            # print("actions", actions.shape, actions[0])
            print(
                # "step:", data["step"],
                "epi_id:", episode_id,
                "epi_step:", episode_step,
                # "action shape:", actions.shape,
                # "action:", actions[0],
                "success:", success_cnt,
                "timeout:", timeout_cnt,
                "terminated:", terminated,
                "truncated:", truncated,
                "object_z:", object_pos[0, 2],
                "ee_z:", actions[:, 2],
                # "action[0,0:3]:", actions[0,0:3],
            )

            if (terminated | truncated).any():
                if episode_id >= num_episodes:
                    break
                
                episode_id += 1
                total_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                
                # reset
                episode_reward = 0.0
                episode_step = 0

                obs_dict, _ = env.reset()
                obs = obs_dict["policy"].to(args_cli.device)
    
    # print summary
    print("-" * 50)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 50)
    print(f"{'Task':<25} | {args_cli.task:<15}")
    print(f"{'Checkpoint':<25} | {Path(checkpoint_path).name:<15}")
    print(f"{'Episodes':<25} | {episode_id:<15}")
    print(f"{'Success Episodes':<25} | {success_cnt:<15}")
    print(f"{'Success Rate':<25} | {success_cnt / episode_id * 100:.1f}%")
    print(f"{'Timeout Episodes':<25} | {timeout_cnt:<15}")
    print(f"{'Timeout Rate':<25} | {timeout_cnt / episode_id * 100:.1f}%")
    print(f"{'Mean Reward':<25} | {sum(total_rewards) / len(total_rewards):.3f}")
    print(f"{'Mean Episode Length':<25} | {sum(episode_lengths) / len(episode_lengths):.1f}")
    if success_steps:
        print(f"{'Mean Success Step':<25} | {sum(success_steps) / len(success_steps):.1f}")
    else:
        print(f"{'Mean Success Step':<25} | N/A")
    print("-" * 50)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

