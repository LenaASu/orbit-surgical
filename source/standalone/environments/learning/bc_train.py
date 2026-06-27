import torch, re
import argparse
from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data.types import Transitions
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

parser = argparse.ArgumentParser(description="Train BC policy using imitation for Isaac Lab environments.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", 
    type=str, 
    default="Isaac-Lift-Needle-PSM-IK-Abs-v0", 
    help="Name of the task."
)


# @dataclasses.dataclass(frozen=True)
# class Trajectory:
#     obs: np.array
#     """Observations, shape(trajectory_len + 1, ) + observation_shape."""
#     acts: np.array
#     """Actions, shape (trajectory_len, ) + action_shape."""
#     infos: Optional[np.array]
#     """An array of info dicts, shape (trajectory_len, )."""
#     terminal: bool
#     """Does this trajectory end in a terminal state?"""


IL_DIR = Path(__file__).resolve().parent
traj_dir = IL_DIR / "data" / "lift_n_trajs_100_v2"
save_path = IL_DIR / "policies" / "bc_lift_n_100_policy_2.zip"

# load trajs
traj_files = sorted(
    traj_dir.glob("lift_n_1_success_ep*.pt"),
    key=lambda p: int(re.search(r"ep(\d+)", p.stem).group(1)),
)

all_obs = []
all_acts = []
all_next_obs = []
all_dones = []
all_infos = []

for traj_file in traj_files:
    traj = torch.load(traj_file, map_location="cpu", weights_only=True)

    for i in range(len(traj) - 1):
        step = traj[i]
        all_obs.append(step["obs"].flatten().numpy())
        all_acts.append(step["action"].flatten().numpy())
        all_next_obs.append(traj[i + 1]["obs"].flatten().numpy())
        all_dones.append(bool(step["terminated"].item() or step["truncated"].item()))
        all_infos.append({})

obs = np.asarray(all_obs, dtype=np.float32)
acts = np.asarray(all_acts, dtype=np.float32)
next_obs = np.asarray(all_next_obs, dtype=np.float32)
dones = np.asarray(all_dones, dtype=bool)
infos = np.asarray(all_infos, dtype=object)

print("obs.shape: ", obs.shape)
print("acts.shape: ", acts.shape)

print("obs min&max: ", obs.min(), obs.max())
print("acts min&max: ", acts.min(), acts.max())

print("obs sum: ", np.isnan(obs).sum())
print("acts sum: ", np.isnan(acts).sum())

print("obs[:5]=", obs[:5])
print("acts[:5]=", acts[:5])

transitions = Transitions(
    obs=obs,
    acts=acts,
    next_obs=next_obs,
    dones=dones,
    infos=infos,
)


rng = np.random.default_rng(0)
action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(8,),
    dtype=np.float32,
)
observation_space = gym.spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(34,),
    dtype=np.float32,
)
# expert = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals-CartPole-v0",
#     venv=env,
# )
# rollouts = rollout.rollout(
#     expert,
#     env,
#     rollout.make_sample_until(min_timesteps=None, min_episodes=50),
#     rng=rng,
# )
# transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=observation_space,
    action_space=action_space,
    demonstrations=transitions,
    rng=rng,
    batch_size=32,
)
bc_trainer.train(n_epochs=2)
bc_trainer.policy.save(str(save_path))
print(bc_trainer.policy)
print(f"Saved BC policy to {save_path}.")
# reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
# print("Reward: ", reward)