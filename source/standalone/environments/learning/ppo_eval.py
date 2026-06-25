"""
Script to run a trained Behavior Cloning policy to pick and lift the suture needle.


.. code-block:: bash

    ${IsaacLab_PATH}/isaaclab.sh -p source/standalone/environments/imitation_learning/ppo_eval.py

    ~/IsaacLab/isaaclab.sh -p source/standalone/environments/imitation_learning/ppo_eval.py 
  
"""

"""Launch Omniverse Toolkit first."""

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
# local imports
import cli_args  # isort: skip

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
    default="Isaac-Lift-Needle-PSM-IK-Abs-Play-v0", 
    help="Name of the task."
)
# parser.add_argument(
#     "--checkpoint", type=str,
#     # default="source/standalone/environments/imitation_learning/policies/bc_lift_n_100_policy_200.pt", 
#     default="/workspace_data/orbit-surgical/logs/rsl_rl/needle_lift/2026-06-24_22-16-35/model_950.pt",
#     help="Pytorch model checkpoint to load."
# )

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.checkpoint is None:
    args_cli.checkpoint = "/workspace_data/orbit-surgical/logs/rsl_rl/needle_lift/2026-06-24_22-16-35/model_950.pt"

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



def main():
    """Play with RSL-RL agent and print results summary."""
    # LOG_DIR = Path("/workspace_data/orbit-surgical/logs/rsl_rl/needle_lift")
    # checkpoint_dir = LOG_DIR / "2026-06-24_22-16-35"

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load data
    # checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    # for checkpoint_file in checkpoint_files:
    #     checkpoint = torch.load(checkpoint_file, map_location=args_cli.device, weights_only=True)

    # checkpoint = torch.load(checkpoint, map_location=args_cli.device, weights_only=True)
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = args_cli.checkpoint
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="2026-06-24_22-16-35", device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(ppo_runner.alg.policy, path=export_model_dir, filename="policy.onnx")


    # reset environment
    obs, info = env.reset()
    
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

     

    while simulation_app.is_running() and episode_id < num_episodes:
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            actions = policy(obs)
           
            obs, rewards, dones, info = env.step(actions)
            # only cares about policy observations
            # obs = obs_dict["policy"].to(args_cli.device)

            # ee_pos = env.unwrapped.scene["robot"].data.body_pos_w[:, ee_body_id]
            # object_pos = env.unwrapped.scene["object"].data.root_pos_w
    

            # episode_reward += rewards.mean().item()
            # episode_step += 1

            success_log = info["log"]["Episode_Termination/object_lifted"]
            timeout_log = info["log"]["Episode_Termination/time_out"]

            if success_log > 0 and dones.any():
                success_cnt += 1
                print(f"Episode {episode_id}: Success")
                # print(info)
            #     success_steps.append(episode_step)
            #     episode_lengths.append(episode_step)
            
            if timeout_log > 0 and dones.any():
                timeout_cnt += 1
                print(f"Episode {episode_id}: Timeout")

            # # print("obs", obs.shape, obs[0, :10])
            # # print("actions", actions.shape, actions[0])
            # print(
            #     # "step:", data["step"],
            #     "epi_id:", episode_id,
            #     "epi_step:", episode_step,
            #     # "action shape:", actions.shape,
            #     # "action:", actions[0],
            #     "success:", success_cnt,
            #     "timeout:", timeout_cnt,
            #     "terminated:", terminated,
            #     "truncated:", truncated,
            #     "object_z:", object_pos[0, 2],
            #     "ee_z:", actions[:, 2],
            #     # "action[0,0:3]:", actions[0,0:3],
            # )

            if dones.any():
                # if episode_id >= num_episodes:
                #     break
                
                episode_id += 1
                # total_rewards.append(episode_reward)
                # episode_lengths.append(episode_step)
                
                # # reset
                # episode_reward = 0.0
                # episode_step = 0
                # print(info)
                # print("success:", success_cnt)
                # print("timeout: ", timeout_cnt)
                obs, info = env.reset()
                # obs = obs_dict["policy"].to(args_cli.device)

    print("-" * 50)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 50)
    print(f"{'Task':<25} | {args_cli.task:<15}")
    print(f"{'Checkpoint':<25} | {Path(checkpoint).name:<15}")
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