import h5py
import torch
from pathlib import Path

def save_demo_to_hdf5(h5_path, episode_traj, demo_id, env_id=0):
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    obs_f = torch.stack([x["obs"][env_id] for x in episode_traj], dim=0).numpy()
    action_f = torch.stack([x["action"][env_id] for x in episode_traj], dim=0).numpy()
    reward_f = torch.stack([x["reward"][env_id] for x in episode_traj], dim=0).numpy()
    # dones_f = torch.cat([x["terminated"] | x["truncated"] for x in episode_traj], dim=0).numpy()

    dones = torch.zeros(len(episode_traj), dtype=torch.bool)
    dones[-1] = True
    dones = dones.numpy()

    with h5py.File(h5_path, "a") as f:
        data_group = f.require_group("data")

        demo_name = f"demo_{demo_id}"
        if demo_name in data_group:
            del data_group[demo_name]

        demo_group = data_group.create_group(demo_name)

        obs_group = demo_group.create_group("obs")
        obs_group.create_dataset("policy", data=obs_f, compression="gzip")

        demo_group.create_dataset("actions", data=action_f, compression="gzip")
        demo_group.create_dataset("rewards", data=reward_f, compression="gzip")
        demo_group.create_dataset("dones", data=dones, compression="gzip")

        demo_group.attrs["num_samples"] = len(episode_traj)
        data_group.attrs["total"] = data_group.attrs.get("total", 0) + len(episode_traj)
