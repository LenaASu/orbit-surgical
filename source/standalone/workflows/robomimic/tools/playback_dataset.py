# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a hdf5 dataset from robomimic."""

"""Launch Isaac Sim Simulator first."""

import argparse,h5py

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play policy trained using robomimic for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="Isaac-Lift-Needle-PSM-IK-Abs-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default="source/standalone/environments/data/datasets/lift_n_dataset_Abs_100.hdf5", help="Hdf5 checkpoint to load.")
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
    # policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    # HDF5
    with h5py.File(args_cli.checkpoint, "r") as f:
        demo = f["data"]["demo_0000"]
        actions = demo["actions"][:]

    obs_dict, _ = env.reset()
    done_reason = "not_done"

    for i, a in enumerate(actions):
        action = torch.tensor(a, device=env.unwrapped.device, dtype=torch.float32).view(1, -1)
        obs_dict, reward, terminated, truncated, info = env.step(action)
        # print("info", info)
        # print("env.unwrapped.termination_manager:", env.unwrapped.termination_manager)
        # print("env.unwrapped.scene.keys():", env.unwrapped.scene.keys())
        success = info["log"]["Episode_Termination/object_lifted"]
        timeout = info["log"]["Episode_Termination/time_out"]
        drop = info["log"]["Episode_Termination/object_dropping"]

      
        needle_z = env.unwrapped.scene["object"].data.root_pos_w[0, 2].item()

        print(
            f"step={i:04d} "
            f"needle_z={needle_z:.5f} "
            f"success={success} "
            f"terminated={terminated} "
            f"truncated={truncated}"
        )

        if success:
            done_reason = "success"
            break
        elif drop:
            done_reason = "drop"
            break
        elif timeout:
            done_reason = "timeout"
            break
        elif terminated or truncated:
            done_reason = "done"
            break

    print(f"\nReplay finished: {done_reason}, last_step={i}, num_actions={len(actions)}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
