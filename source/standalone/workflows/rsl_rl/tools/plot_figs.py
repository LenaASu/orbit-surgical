import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROBOMIMIC_DIR = TOOLS_DIR.parent

RESULT_DIR = ROBOMIMIC_DIR / "results" 
CSV_DIR = RESULT_DIR / "tb_csv"
save_path_loss = RESULT_DIR / "figs" / "loss_lr.png"
save_path_lift = RESULT_DIR / "figs" / "object_lifted.png"
save_path_reward = RESULT_DIR / "figs" / "train_mean_reward.png"

import pandas as pd
import matplotlib.pyplot as plt

def plot_loss(df, save_path, smoothing=0.6):
    steps = df["Step"]
    values = df["Value"]
    smoothed = values.ewm(alpha=1 - smoothing, adjust=False).mean()

    plt.figure(figsize=(8, 5))

    # raw data: light
    plt.plot(
        steps,
        values,
        color="#7DAEE0",
        linewidth=1,
        alpha=0.4,
        label="Raw",
    )

    # smoothed data: dark
    plt.plot(
        steps,
        smoothed,
        color="#4988C7",
        linewidth=1,
        label="Smoothed",
    )

    plt.title("Learning Rate", fontsize=14)
    # plt.xlim(0, 200)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_lift(df, save_path_lift, smoothing=0.6):
    steps = df["Step"]
    values = df["Value"]
    smoothed = values.ewm(alpha=1 - smoothing, adjust=False).mean()

    plt.figure(figsize=(8, 5))

    # raw data: light
    plt.plot(
        steps,
        values,
        color="#7DAEE0",
        linewidth=1,
        alpha=0.4,
        label="Raw",
    )

    # smoothed data: dark
    plt.plot(
        steps,
        smoothed,
        color="#4988C7",
        linewidth=1,
        label="Smoothed",
    )

    plt.title("Mean Successful Terminations per Rollout", fontsize=14)
    # plt.xlim(0, 200)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Success Terminations", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path_lift, dpi=300, bbox_inches="tight")
    plt.close()

def plot_reward(df, save_path_reward):
    plt.figure(figsize=(8, 5))
    plt.plot(df['Step'], df['Value'], label='Reward', color="#4988C7", linewidth=1)

    plt.title("Mean Training Reward", fontsize=14)
    # plt.xlim(0, 200)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path_reward, dpi=300, bbox_inches="tight")
    plt.close()

def main():
   
    train_loss_df = pd.read_csv(CSV_DIR / "loss_lr.csv")
    object_lifted_df = pd.read_csv(CSV_DIR / "object_lifted.csv")
    reward_df = pd.read_csv(CSV_DIR / "train_mean_reward.csv")

    save_path_loss.parent.mkdir(parents=True, exist_ok=True)
    save_path_lift.parent.mkdir(parents=True, exist_ok=True)
    save_path_reward.parent.mkdir(parents=True, exist_ok=True)

    plot_loss(train_loss_df, save_path_loss)
    plot_lift(object_lifted_df, save_path_lift)
    plot_reward(reward_df, save_path_reward)

    print(f"Saved all figs.")

if __name__ == "__main__":
    main()
    