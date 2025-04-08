# plot_lr_ablation.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Config
data_dir = "data"
file_prefix = "actor_critic_lr_"
file_suffix = ".csv"

# Extract all matching filenames
all_files = [f for f in os.listdir(data_dir) if f.startswith(file_prefix) and f.endswith(file_suffix)]

# Extract learning rate from filename using regex
def extract_lr(filename):
    match = re.search(r"lr_(\d+e?-?\d*)", filename)
    return float(match.group(1)) if match else None

# Plot setup
sns.set(style="whitegrid", palette="colorblind")
plt.figure(figsize=(10, 6))

for file in sorted(all_files, key=extract_lr):
    lr = extract_lr(file)
    if lr is None:
        continue

    path = os.path.join(data_dir, file)
    df = pd.read_csv(path)

    # Smooth with a 50-episode rolling average
    smooth_window = 100
    smoothed = df["Return"].rolling(window=smooth_window, min_periods=1).mean()

    # Downsample: plot every Nth point based on data length
    downsample_step = max(len(df) // 100, 1)
    
    plt.plot(
        df["Episode"].iloc[::downsample_step],
        smoothed.iloc[::downsample_step],
        label=f"LR={lr:.0e}"
    )

# Final plot labels and save
plt.title("Ablation Study: Learning Rate on Actor-Critic (CartPole)")
plt.xlabel("Episode")
plt.ylabel("Smoothed Return (Window=50)")
plt.legend(title="Learning Rate", loc="lower right")
plt.grid(True)

# Save and show
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/actor_critic_lr_ablation.png", dpi=300)
plt.show()
