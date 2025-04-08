# plot_critic_capacity.py
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Settings ---
data_dir = "data"
save_path = "plots/critic_capacity_ablation.png"
rolling_window = 200
max_points = 5  # Max number of points to plot per curve

# --- Plot config ---
sns.set(style="whitegrid", palette="colorblind")
plt.figure(figsize=(10, 6))

# --- File matching ---
pattern = re.compile(r"actor_critic_dims_\[?([0-9, ]+)\]?\.csv")

# --- Process each file ---
for filename in os.listdir(data_dir):
    match = pattern.match(filename)
    if not match:
        continue

    dims = match.group(1).replace(",", ", ")  # Prettify
    label = f"[{dims}]"
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)

    # Smooth returns
    smoothed = df["Return"].rolling(window=rolling_window, min_periods=1).mean()

    # Downsample for clarity
    downsample = max(len(df) // max_points, 1)
    plt.plot(df["Episode"].iloc[::downsample], smoothed.iloc[::downsample], label=label)

# --- Final plot styling ---
plt.title("Critic Network Capacity Ablation (Actor-Critic)")
plt.xlabel("Episode")
plt.ylabel(f"Smoothed Return (Window={rolling_window})")
plt.legend(title="Critic Layers", loc="lower right")
plt.grid(True)
plt.tight_layout()

# --- Save and show ---
os.makedirs("plots", exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.show()
