import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Settings
data_dir = "data"
file_prefix = "actor_critic_actorlr_"
file_suffix = ".csv"
smooth_window = 50
max_points = 200

# File parser
def extract_lrs(filename):
    match = re.search(r"actorlr_(\d+e?-?\d*)_criticlr_(\d+e?-?\d*)", filename)
    if match:
        actor_lr = float(match.group(1))
        critic_lr = float(match.group(2))
        return actor_lr, critic_lr
    return None, None

# Scan files
files = [
    f for f in os.listdir(data_dir)
    if f.startswith(file_prefix) and f.endswith(file_suffix)
]

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(12, 7))

for file in sorted(files):
    actor_lr, critic_lr = extract_lrs(file)
    if actor_lr is None or critic_lr is None:
        continue

    path = os.path.join(data_dir, file)
    df = pd.read_csv(path)

    # Smooth and downsample
    smoothed = df["Return"].rolling(window=smooth_window, min_periods=1).mean()
    downsample_step = max(len(df) // max_points, 1)

    plt.plot(
        df["Episode"].iloc[::downsample_step],
        smoothed.iloc[::downsample_step],
        label=f"A:{actor_lr:.0e}, C:{critic_lr:.0e}"
    )

# Final touches
plt.title("Actor-Critic: Actor vs Critic LR Ablation (CartPole)")
plt.xlabel("Episode")
plt.ylabel(f"Smoothed Return (Window={smooth_window})")
plt.legend(title="Actor LR, Critic LR", loc="lower right", fontsize='small')
plt.grid(True)

# Save & show
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/actor_vs_critic_lr_ablation.png", dpi=300)
plt.show()
