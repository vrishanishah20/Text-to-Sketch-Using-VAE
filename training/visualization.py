import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

log_dir = "lightning_logs"

# find latest run
latest_version = sorted(os.listdir(log_dir))[-1]
events_path = os.path.join(log_dir, latest_version)

# Load TensorBoard logs
ea = event_accumulator.EventAccumulator(events_path)
ea.Reload()

# loss logs
train_loss = ea.Scalars("train_loss")
val_loss = ea.Scalars("val_loss")

# DataFrames
df_train = pd.DataFrame([(x.step, x.value) for x in train_loss], columns=["step", "train_loss"])
df_val = pd.DataFrame([(x.step, x.value) for x in val_loss], columns=["step", "val_loss"])


df = pd.merge(df_train, df_val, on="step", how="outer").sort_values("step")

plt.figure(figsize=(10, 5))
plt.plot(df["step"], df["train_loss"], label="Train Loss", marker='o')
plt.plot(df["step"], df["val_loss"], label="Validation Loss", marker='x')
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()
