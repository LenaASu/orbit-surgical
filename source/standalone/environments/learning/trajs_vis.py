import os
import math
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# load trajectory
# os.makedirs("results", exist_ok=True)

FILE_NAME = "lift_n_1_200_bc_policy"
num_envs = 1
DECIMATION = 4
SIM_DT = 1.0 / 200.0
EPISODE_LENGTH_S = 8.0
traj = torch.load(f"{FILE_NAME}.pt", map_location="cpu", weights_only=True)

print(type(traj))
print(traj)
# extract params
ee = torch.cat([x["ee_pos"] for x in traj], dim=0).numpy()
object = torch.cat([x["object_pos"] for x in traj], dim=0).numpy()
steps = [x["step"] for x in traj]
rewards = [x["reward"].float().mean().item() for x in traj]
# terminated = [x["terminated"].item() for x in traj]
# truncated = [x["truncated"].item() for x in traj]
success = [x["object_lifted_log"] for x in traj]
timeout = [x["timeout_log"] for x in traj]

# print(type(rewards))
# print(rewards)
# print(type(success))
# print(success)


# summary
epi_length = EPISODE_LENGTH_S / (SIM_DT * DECIMATION)
first_success_idx = -1
success_episodes = 0
success_reward_sum = 0
timeout_episode = 0
for i in range(len(success)):
    if success[i] == num_envs:
        if first_success_idx == -1:
            first_success_idx = i
        success_episodes += 1
        success_reward_sum += rewards[i]
episode_success_rate = success_episodes / len(success)
if success_episodes != 0:
    success_reward_avg = success_reward_sum / success_episodes
else:
    success_reward_avg = 0

episodes = len(success)/epi_length
timeout_episodes = math.ceil(sum(timeout) / epi_length)
episode_timeout_rate = timeout_episodes / episodes
# print(timeout)

task_name = FILE_NAME.split('_', 2)[1]
task = ''
if task_name == 'n':
    task = 'Lift Needle'
if task_name == 'b':
    task = 'Lift Block'

header = ["Metric", "Value"]
table_data = [
    ["Task", task],
    ["Num Envs", num_envs],
    ["Episodes", f"{episodes:.0f}"],
    ["Episode Length", f"{epi_length:.0f} steps"],
    ["Total Steps", f"{len(success):.0f}"],
    ["Total Reward", f"{sum(rewards):.3f}"],
    ["Mean Reward", f"{(sum(rewards)/len(rewards)):.3f}"],
    ["Timeout Episodes", f"{timeout_episodes:.0f}"],
    ["Timeout Rate", f"{(episode_timeout_rate) * 100:.1f}%"]

]

if first_success_idx != -1:
    table_data.append(["First Success Step", first_success_idx])
    table_data.append(["Success Episodes", f"{(success_episodes/epi_length):.0f}"])
    table_data.append(["Success Rate", f"{episode_success_rate * 100:.1f}%"])
    table_data.append(["Mean Success Reward", f"{success_reward_avg:.3f}"])
else:
    table_data.append(["Success Status", "No perfect success"])

print("-" * 45)
print(f"{header[0]:<25} | {header[1]:<15}")
print("-" * 45)
for row in table_data:
    print(f"{row[0]:<25} | {row[1]:<15}")
print("-" * 45)



# 2D top view
plt.figure(figsize=(6, 6))

plt.plot(ee[:, 0], ee[:, 1], label="EE Trajectory")
plt.plot(object[:, 0], object[:,1 ], label="object Trajectory")

plt.scatter(ee[0, 0], ee[0, 1], marker="o", label="EE Start")
plt.scatter(ee[-1, 0], ee[-1, 1], marker="x", label="EE End")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Top View Trajectory")
plt.legend()
plt.axis("equal")

plt.savefig(f"results/{FILE_NAME}_top_view.png", dpi=300, bbox_inches="tight")
plt.close()


# 3D trajectory
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    ee[:,0],
    ee[:,1],
    ee[:,2],
    label="EE"
)

ax.plot(
    object[:,0],
    object[:,1],
    object[:,2],
    label="object"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("3D Trajectory")
ax.legend()

plt.savefig(f"results/{FILE_NAME}_traj_3d.png", dpi=300, bbox_inches="tight")
plt.close()


# object lifting curve
plt.figure(figsize=(8,4))

plt.plot(object[:,2])

plt.xlabel("Step")
plt.ylabel("object Height (Z)")
plt.title("object Lifting")

plt.savefig(f"results/{FILE_NAME}_obj_h.png", dpi=300, bbox_inches="tight")
plt.close()


# reward curve
plt.figure(figsize=(8,4))

plt.plot(steps, rewards)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Curve")

plt.savefig(f"results/{FILE_NAME}_rewards.png", dpi=300, bbox_inches="tight")
plt.close()


# # terminated
# plt.figure(figsize=(8,4))

# plt.plot(steps, terminated)
# plt.xlabel("Step")
# plt.ylabel("Terminated")
# plt.title("Terminated Curve")

# plt.savefig(f"results/{FILE_NAME}_term.png", dpi=300, bbox_inches="tight")
# plt.close()

# # truncated
# plt.figure(figsize=(8,4))

# plt.plot(steps, truncated)
# plt.xlabel("Step")
# plt.ylabel("Truncated")
# plt.title("Truncated Curve")

# plt.savefig(f"results/{FILE_NAME}_trun.png", dpi=300, bbox_inches="tight")
# plt.close()

# success
plt.figure(figsize=(8,4))

plt.plot(steps, success)
plt.xlabel("Step")
plt.ylabel("Success")
plt.title("Success Curve")

plt.savefig(f"results/{FILE_NAME}_success.png", dpi=300, bbox_inches="tight")
plt.close()

# interactive 3D trajectory
fig_html = go.Figure()

fig_html.add_trace(
    go.Scatter3d(
        x=ee[:, 0],
        y=ee[:, 1],
        z=ee[:, 2],
        mode="lines+markers",
        name="EE"
    )
)

fig_html.add_trace(
    go.Scatter3d(
        x=object[:, 0],
        y=object[:, 1],
        z=object[:, 2],
        mode="lines+markers",
        name="object"
    )
)

fig_html.update_layout(
    title="Interactive 3D Trajectory",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    )
)

fig_html.write_html(f"results/{FILE_NAME}_traj_3d.html")


print(f"Saved results for {FILE_NAME}.")
