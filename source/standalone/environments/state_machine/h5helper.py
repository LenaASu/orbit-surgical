import h5py, json
import torch
import pandas as pd
from pathlib import Path
import numpy as np

def save_demo_to_hdf5(h5_path, episode_traj, demo_id, task):
    if len(episode_traj) == 0:
        return
    
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    action_f = torch.stack([x["action"] for x in episode_traj], dim=0).numpy()
    reward_f = torch.stack([x["reward"] for x in episode_traj], dim=0).numpy()
    # dones_f = torch.cat([x["terminated"] | x["truncated"] for x in episode_traj], dim=0).numpy()

    dones = torch.zeros(len(episode_traj), dtype=torch.bool)
    dones[-1] = True
    dones = dones.numpy()

    with h5py.File(h5_path, "a") as f:
        data_group = f.require_group("data")

        demo_name = f"demo_{demo_id:04d}"
        if demo_name in data_group:
            del data_group[demo_name]

        demo_group = data_group.create_group(demo_name)
        obs_group = demo_group.create_group("obs")
        obs_dim = 0

        obs_keys = episode_traj[0]["obs"]["policy"].keys()
        # obs_group.create_dataset("policy", data=obs_f, compression="gzip")
        for key in obs_keys:
            obs_f = torch.stack([x["obs"]["policy"][key] for x in episode_traj], dim=0).cpu().numpy()
            obs_group.create_dataset(key, data=obs_f, compression="gzip")
            obs_dim += obs_f.shape[-1]

        demo_group.create_dataset("actions", data=action_f, compression="gzip")
        demo_group.create_dataset("rewards", data=reward_f, compression="gzip")
        demo_group.create_dataset("dones", data=dones, compression="gzip")
        
        data_group.attrs["num_demos"] = len(data_group)
        demo_group.attrs["num_samples"] = len(episode_traj)
        data_group.attrs["total"] = sum(data_group[k].attrs["num_samples"] for k in data_group.keys())
        
        data_group.attrs["collector"] = "sm" # state machine

        data_group.attrs["obs_dim"] = obs_dim
        data_group.attrs["action_dim"] = action_f.shape[-1]
        data_group.attrs["env_args"] = json.dumps({
            "env_name": task,
            "type": 2,
            "env_kwargs": {},
        })

def analyze_h5(h5_path, save_path):
    with h5py.File(h5_path, 'r') as file:
        print(f"Keys: {list(file.keys())}")
        a_group_key = list(file.keys())[0]

        data = list(file[a_group_key])
        print(data)

def print_h5_structure(group, indent=0):
    for key in group.keys():
        item = group[key]
        print(" " * indent + key)
        if isinstance(item, h5py.Group):
            print_h5_structure(item, indent + 1)
        else:
            print(" " * (indent + 1) + f"shape={item.shape}")

def print_h5_summary(h5_path):
    with h5py.File(h5_path, "r") as f:
        data = f["data"]

        lengths = [data[k].attrs["num_samples"] for k in data.keys()]
        num_demos = len(lengths)
        total_samples = sum(lengths)

        print("\n" + "=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        print(f"Number of demos     : {num_demos}")
        print(f"Total samples       : {total_samples}")
        print(f"Mean traj length    : {np.mean(lengths):.1f}")
        print(f"Std traj length     : {np.std(lengths):.1f}")
        print(f"Median traj length  : {np.median(lengths):.1f}")
        print(f"Min traj length     : {min(lengths)}")
        print(f"Max traj length     : {max(lengths)}")

        print(f"Observation dim     : {data.attrs['obs_dim']}")
        print(f"Action dim          : {data.attrs['action_dim']}")
        print(f"Collector           : {data.attrs['collector']}")
        print(f"Environment         : {json.loads(data.attrs['env_args'])['env_name']}")
        print("=" * 60)
