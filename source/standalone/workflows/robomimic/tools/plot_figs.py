import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROBOMIMIC_DIR = TOOLS_DIR.parent

RESULT_DIR = ROBOMIMIC_DIR / "results" / "bc_200demo_50epo" / "tb_csv"
save_path = RESULT_DIR / "figs" / "loss.png"
save_path_grad = RESULT_DIR / "figs" / "policy_grad_norm.png"

def plot_loss(train_df, valid_df, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_df['Step'], train_df['Value'], label='Training Loss', color="#7DAEE0", linewidth=2)
    plt.plot(valid_df['Step'], valid_df['Value'], label='Validation Loss', color="#D4666F", linewidth=2)

    plt.title("Training and Validation Loss", fontsize=14)
    plt.xlim(0, 200)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_policy_grad_norm(df, save_path_grad):
    plt.figure(figsize=(8, 5))
    plt.plot(df['Step'], df['Value'], label='Policy Grad Norm', color="#7DAEE0", linewidth=2)

    plt.title("Training Policy Grad Norms", fontsize=14)
    plt.xlim(0, 200)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Grad Norms", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path_grad, dpi=300, bbox_inches="tight")
    plt.close()

def main():
   
    train_loss_df = pd.read_csv(RESULT_DIR / "train_loss.csv")
    valid_loss_df = pd.read_csv(RESULT_DIR / "valid_loss.csv")
    norm_df = pd.read_csv(RESULT_DIR / "train_policy_grad_norm.csv")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_loss(train_loss_df, valid_loss_df, save_path)
    plot_policy_grad_norm(norm_df, save_path_grad)

    print(f"Saved files to {save_path}")

if __name__ == "__main__":
    main()
    