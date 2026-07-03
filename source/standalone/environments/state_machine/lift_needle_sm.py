# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ${IsaacLab_PATH}/isaaclab.sh -p source/standalone/environments/state_machine/lift_needle_sm.py --num_envs 1

"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Needle-PSM-IK-Abs-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
# app_launcher = AppLauncher(headless=args_cli.headless)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
<<<<<<< HEAD
from h5helper import save_demo_to_hdf5, print_h5_summary
=======
from tohdf5 import save_demo_to_hdf5
>>>>>>> 2622ce1 (Save as h5)
from collections.abc import Sequence

import warp as wp

from isaaclab.assets import RigidObject
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

from isaaclab_tasks.utils import parse_env_cfg

from isaaclab.utils.math import subtract_frame_transforms

import orbit.surgical.tasks  # noqa: F401
from orbit.surgical.tasks.surgical.lift.lift_env_cfg import LiftEnvCfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.5)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(0.7)
    GRASP_OBJECT = wp.constant(0.8)
    LIFT_OBJECT = wp.constant(2.0)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0

    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        # self.offset[:, 2] = 0.05
        self.offset[:, 2] = 0.01
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.observations.policy.concatenate_terms = False
    # create environment
<<<<<<< HEAD
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
=======
    raw_env = gym.make("Isaac-Lift-Needle-PSM-IK-Abs-v0", cfg=env_cfg)
>>>>>>> 2622ce1 (Save as h5)

    # record video
    # raw_env = gym.wrappers.RecordVideo(
    #     raw_env,
    #     video_folder="./videos",
    #     episode_trigger=lambda episode_id:True
    #     )
    
    base_env = raw_env.unwrapped
    # reset environment at start
    raw_env.reset()
    base_env.sim.step()

    # create action buffers (position + quaternion)
    # actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    # actions[:, 3] = 1.0

    # # create state machine
    # pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    actions = torch.zeros(base_env.action_space.shape, device=base_env.device)
    actions[:, 3] = 1.0

    pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, base_env.num_envs, base_env.device)

    # Create traj set
    episode_traj = []
    episode_id = 1
    episode_step = 0
    # episode_length = env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)
    episode_length = base_env.max_episode_length
    
    step_cnt = 0
    success_log = 0
    success_cnt = 0
    timeout_log = 0
    timeout_cnt = 0
    drop_log = 0
    drop_cnt = 0
    num_episodes = 3 # set number of episodes
<<<<<<< HEAD
    target_success = 50 # set number of successful trajs
=======
    target_success = 10 # set number of successful trajs
>>>>>>> 2622ce1 (Save as h5)
    episode_saved = False # init bool for saving traj
    # max_steps = num_episodes * episode_length
    
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # executed_actions = actions.clone()

            # step environment
            obs_dict, reward, terminated, truncated, info = raw_env.step(actions)
            dones = terminated | truncated

            # observations
            robot: RigidObject = base_env.scene["robot"]
            # -- end-effector frame
            ee_frame_sensor = base_env.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - base_env.scene.env_origins
            tcp_rest_position_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], tcp_rest_position
            )
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            # -- object frame
            object_data: RigidObjectData = base_env.scene["object"].data
            object_position = object_data.root_pos_w - base_env.scene.env_origins
            object_position_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_position
            )
            object_orientation = object_data.root_quat_w
            # -- target object frame
            # desired_pose = env.unwrapped.command_manager.get_command("object_pose")
            desired_pose = base_env.command_manager.get_command("object_pose")


            # Add traj
            # if step_cnt % 5 == 0:
            episode_traj.append({
                "step": step_cnt,
                "episode_id": episode_id,
                "sm_state": pick_sm.sm_state.detach().cpu(),

<<<<<<< HEAD
                "obs": {
                    "policy": {
                        k: v.detach().cpu() for k, v in obs_dict["policy"].items()
                    }
                },
=======
                "obs": obs_dict["policy"].cpu(),
>>>>>>> 2622ce1 (Save as h5)
                "action": actions.detach().cpu(),
                "reward": reward.detach().cpu(),
                "ee_pos": tcp_rest_position.detach().cpu(),
                "object_pos": object_position.detach().cpu(),
                
                "terminated": terminated.detach().cpu(),
                "truncated": truncated.detach().cpu(),
                "object_lifted_log": success_log,
                "timeout_log": timeout_log,
            })

<<<<<<< HEAD
            save_path = f"source/standalone/environments/data/datasets/lift_n_dataset_Abs_50_v3.hdf5"
            
=======
>>>>>>> 2622ce1 (Save as h5)
            if dones.any():
                success_log = info["log"]["Episode_Termination/object_lifted"]
                timeout_log = info["log"]["Episode_Termination/time_out"]
                drop_log = info["log"]["Episode_Termination/object_dropping"]

                if success_log == 1:
                    success_cnt += 1
                   
<<<<<<< HEAD
                    save_demo_to_hdf5(save_path, episode_traj, success_cnt - 1, args_cli.task)
=======
                    save_path = f"source/standalone/environments/data/lift_n_trajs_100_v3/lift_n_1_dataset_Abs.hdf5"
                    save_demo_to_hdf5(save_path, episode_traj, success_cnt - 1)
>>>>>>> 2622ce1 (Save as h5)
                    print(f"Saved demo_{success_cnt - 1}")
                    # episode_saved = True

                if timeout_log == 1:
                    timeout_cnt += 1

                if drop_log == 1:
                    drop_cnt += 1

                episode_traj = []
                episode_step = 0
                episode_id += 1

<<<<<<< HEAD
                if success_cnt >= target_success:
                    print_h5_summary(save_path)
                    break

                # reset
                raw_env.reset()
                base_env.sim.step()
                pick_sm.reset_idx()
                actions = torch.zeros(base_env.action_space.shape, device=base_env.device)
                actions[:, 3] = 1.0
                
                continue
=======
                raw_env.reset()
                base_env.sim.step()
>>>>>>> 2622ce1 (Save as h5)

            # advance state machine
            actions = pick_sm.compute(
                torch.cat([tcp_rest_position_b, tcp_rest_orientation], dim=-1),
                torch.cat([object_position_b, object_orientation], dim=-1),
                desired_pose,
                
            )
            step_cnt += 1
            episode_step += 1

            if step_cnt % 10 == 0:
                print("step: ", step_cnt)
                print("episode: ", episode_id)
                print("success_cnt: ", success_cnt)
                print("timeout_cnt: ", timeout_cnt)
                print("drop_cnt: ", drop_cnt)
                # print("obs_dict_policy: ", obs_dict["policy"])
                # print("ee pose input[0]: ", torch.cat([tcp_rest_position_b, tcp_rest_orientation], dim=-1))
                # print("object pose input[0]: ", torch.cat([object_position_b, object_orientation], dim=-1))
                # print("desired object pose[0]: ", desired_pose[0])
                # print("actions shape: ", actions.shape)
                # print("actions[0]: ", actions[0])
                print("sm_state: ", pick_sm.sm_state)
                # print("gripper: ", actions[0, 7])
                # print("object_z: ", object_position_b[0, 2])
                # print("ee_z: ", tcp_rest_position_b[0, 2])
                # print("success: ", success)
                print("terminated: ", terminated)
                print("truncated: ", truncated)
                # print(info)

           
    # close the environment
    raw_env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
