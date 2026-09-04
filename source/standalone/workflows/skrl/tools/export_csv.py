import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVENT_FILE = (
    "logs/skrl/lift/2026-08-18_13-36-19_td3_torch/"
    "events.out.tfevents.1787078181.lena-Precision-3440.9567.0"
)

OUTPUT_FILE = "source/standalone/workflows/skrl/results/tb_csv/td3_tensorboard_scalars.csv"

# Load TensorBoard event file
event_acc = EventAccumulator(EVENT_FILE)
event_acc.Reload()

# Print available scalar tags
scalar_tags = event_acc.Tags()["scalars"]

print("Scalar tags:")
for tag in scalar_tags:
    print(f"  {tag}")

# Export all scalar data
rows = []

for tag in scalar_tags:
    events = event_acc.Scalars(tag)

    for event in events:
        rows.append(
            {
                "tag": tag,
                "step": event.step,
                "value": event.value,
                "wall_time": event.wall_time,
            }
        )

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nExported {len(df)} scalar records")
print(f"Saved to: {OUTPUT_FILE}")