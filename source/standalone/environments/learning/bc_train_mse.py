from pathlib import Path
import argparse, torch, glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

class BCPolicy(nn.Module):
    def __init__(self, obs_dim=34, act_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),   
        )

    def forward(self, obs):
        return self.net(obs)

def train():
    parser = argparse.ArgumentParser(description="Train BC policy for Isaac Lab environments.")
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="Isaac-Lift-Needle-PSM-IK-Abs-Play-v0", 
        help="Name of the task."
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    IL_DIR = Path(__file__).resolve().parent
    traj_dir = IL_DIR / "data" / "lift_n_trajs_100_v2"
    save_path = IL_DIR / "policies" / "bc_lift_n_100_policy_200.pt"

    all_obs = []
    all_acts = []

    traj_files = sorted(traj_dir.glob("*.pt"))
    for traj_file in traj_files:
        traj = torch.load(traj_file, map_location="cpu", weights_only=True)

        for step in traj[:-1]:
            all_obs.append(step["obs"].flatten())
            all_acts.append(step["action"].flatten()[:3])  # only learn xyz

    obs = torch.stack(all_obs).float()
    acts = torch.stack(all_acts).float()
    # obs = np.asarray(all_obs, dtype=np.float32)
    # acts = np.asarray(all_acts, dtype=np.float32)
    print("obs shape:", obs.shape)
    print("acts shape:", acts.shape)

    dataset = TensorDataset(obs, acts)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = BCPolicy(obs_dim=34, act_dim=3).to(device)
    optimizer = Adam(policy.parameters(), lr=1e-3)
    
    for epoch in range(args.epochs):
        total_loss = 0.0

        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred = policy(obs_batch)
            loss = F.mse_loss(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * obs_batch.shape[0]

        mean_loss = total_loss/len(dataset)

        if epoch % 10 ==0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:04d} | mse loss: {mean_loss:.6f}")

    torch.save({"model_state_dict": policy.state_dict(),
                "obs_dim": 34,
                "act_dim": 3,
                }, 
                save_path,
            )

    print(f"Saved policy to {save_path}")

if __name__ == "__main__":
    train()