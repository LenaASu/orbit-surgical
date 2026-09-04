import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE_PATH = "logs/skrl/lift/2026-08-18_13-36-19_td3_torch/td3_trajectories_0818.csv"
SAVE_PATH = Path("source/standalone/workflows/skrl/results/figs")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(FILE_PATH)

# x = data['step']
# y1 = data['ee_object_distance']

# plt.plot(x, y1)
# plt.show()

success = data[
        (data["checkpoint"] == "agent_33600.pt") &
        (data["episode"] == 42)
    ]

failure = data[
    (data["checkpoint"] == "agent_33600.pt") &
    (data["episode"] == 32)
]

def plot_success_failure(success, failure):
    plt.plot(
        success["step"],
        success["ee_object_distance"],
        label="Success (Episode 42)"
    )

    plt.plot(
        failure["step"],
        failure["ee_object_distance"],
        label="Failure (Episode 32)"
    )

    plt.xlabel("Step")
    plt.ylabel("EE-Object Distance (m)")
    plt.title("TD3 Success vs Failure")
    plt.legend()
    plt.grid(True)

    plt.savefig(SAVE_PATH / "success_failure.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    
def plot_gripper_action(success, failure):
    plt.plot(
        success["step"],
        success["gripper"],
        label="Success"
    )

    plt.plot(
        failure["step"],
        failure["gripper"],
        label="Failure"
    )

    plt.xlabel("Step")
    plt.ylabel("Gripper Action")
    plt.title("TD3 Gripper Action: Success vs Failure")
    plt.xlim(0, 30)
    plt.legend()
    plt.grid(True)

    plt.savefig(SAVE_PATH / "gripper_action.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_all_dim(failure):
    for i in range(7):
        plt.figure()
        plt.plot(failure["step"], failure[f"action_{i}"])
        plt.xlabel("Step")
        plt.ylabel(f"Action {i}")
        plt.title(f"Failure Episode - Action {i}")
        plt.grid(True)
        plt.savefig(SAVE_PATH / f"gripper_action_{i}.png", dpi=300, bbox_inches="tight")
        # plt.show()

# plot_success_failure(success, failure)
# plot_gripper_action(success, failure)
plot_all_dim(failure)