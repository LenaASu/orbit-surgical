# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""

<<<<<<< HEAD
import argparse,h5py
=======
import argparse
>>>>>>> 71eec3b (Fixed API)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using robomimic for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import robomimic  # noqa: F401
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import isaaclab.app  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import orbit.surgical.tasks  # noqa: F401


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    # we want to have the terms in the observations returned as a dictionary
    # rather than a concatenated tensor
    env_cfg.observations.policy.concatenate_terms = False

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

<<<<<<< HEAD
    # HDF5
    # with h5py.File(args_cli.checkpoint, "r") as f:
    #     demo = f["data"]["demo_0000"]
    #     actions = demo["actions"][:]

    # obs_dict, _ = env.reset()

    # for i, a in enumerate(actions):
    #     action = torch.tensor(a, device=env.device, dtype=torch.float32).view(1, -1)
    #     obs_dict, reward, terminated, truncated, info = env.step(action)

    #     success = info["log"]["Episode_Termination/object_lifted"]
    #     timeout = info["log"]["Episode_Termination/time_out"]
    #     drop = info["log"]["Episode_Termination/object_dropping"]

    #     print(i)
    #     # print(i, "success", success, "timeout", timeout, "drop", drop, "reward", reward)

    #     if success or timeout or drop or terminated or truncated:
    #         print(i, "success", success, "timeout", timeout, "drop", drop, "reward", reward)
    #         break

    # BC eval
    # reset environment
    obs_dict, _ = env.reset()

    step = 0
    episode_step = 0
    episode_id = 1
    success_cnt = 0
    timeout_cnt = 0
    drop_cnt = 0

    num_episodes = 50
=======
    # reset environment
    obs_dict, _ = env.reset()

    step = 0
    episode_step = 0
    episode_id = 1
    success_cnt = 0
    timeout_cnt = 0
    drop_cnt = 0

<<<<<<< HEAD

>>>>>>> 71eec3b (Fixed API)
=======
    num_episodes = 50
>>>>>>> 913edfe (Robomimic BC works)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 913edfe (Robomimic BC works)
            # actions = policy(obs)
            # actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

            policy_obs = obs_dict["policy"]

            robomimic_obs = {
                "joint_pos": policy_obs["joint_pos"],
                "joint_vel": policy_obs["joint_vel"],
                "object_position": policy_obs["object_position"],
                "target_object_position": policy_obs["target_object_position"],
            }

            action = policy(robomimic_obs)
            if not isinstance(action, torch.Tensor):
                action = torch.from_numpy(action)
            action = action.to(device=env.device).view(1, -1)

            demo_action = policy_obs["actions"]
            pred_action = action
            print("pred:", pred_action.cpu().numpy())
            print("demo:", demo_action.cpu().numpy())
            print("MAE:", torch.mean(torch.abs(pred_action - demo_action)).item())

<<<<<<< HEAD
            # apply actions
            obs_dict, reward, terminated, truncated, info = env.step(action)
            step += 1
            episode_step += 1
=======
            actions = policy(obs)
            actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
=======
>>>>>>> 913edfe (Robomimic BC works)
            # apply actions
            obs_dict, reward, terminated, truncated, info = env.step(action)
            step += 1
<<<<<<< HEAD
            # robomimic only cares about policy observations
            obs = obs_dict["policy"]
>>>>>>> 71eec3b (Fixed API)
=======
            episode_step += 1
>>>>>>> 913edfe (Robomimic BC works)

            success = info["log"]["Episode_Termination/object_lifted"]
            timeout = info["log"]["Episode_Termination/time_out"]
            drop = info["log"]["Episode_Termination/object_dropping"]

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 913edfe (Robomimic BC works)
            print(f"success={success_cnt}, timeout={timeout_cnt}, drop={drop_cnt}")
            print("episode_step", episode_step)
            print("episode", episode_id)
            print("action", action.cpu().numpy())
<<<<<<< HEAD
            print("reward", reward)
            print("object_pos", policy_obs["object_position"].cpu().numpy())

            if (terminated | truncated):
                if success:
                    success_cnt += 1
                if timeout:
                    timeout_cnt += 1
                if drop:
                    drop_cnt += 1
                print("Episode:", episode_id)
                episode_id += 1
                episode_step = 0

            if episode_id > num_episodes:
                print("Final success:", success_cnt)
                print("Success_rate:", success_cnt / num_episodes)
=======
            print(f"success={success}, timeout={timeout}, drop={drop}")
            print("step", step)
            print("action", actions.cpu().numpy())
=======
>>>>>>> 913edfe (Robomimic BC works)
            print("reward", reward)
            print("object_pos", policy_obs["object_position"].cpu().numpy())

<<<<<<< HEAD
            if success or timeout or drop:
>>>>>>> 71eec3b (Fixed API)
=======
            if (terminated | truncated):
                if success:
                    success_cnt += 1
                if timeout:
                    timeout_cnt += 1
                if drop:
                    drop_cnt += 1
                print("Episode:", episode_id)
                episode_id += 1
                episode_step = 0

            if episode_id > num_episodes:
                print("Final success:", success_cnt)
                print("Success_rate:", success_cnt / num_episodes)
>>>>>>> 913edfe (Robomimic BC works)
                break
            

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
