import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# File paths
baseline_paths = {
    "REINFORCE": "data/reinforce_returns.csv",
    "Actor-Critic (AC)": "data/actor_critic_returns.csv",
    "Advantage Actor-Critic (A2C)": "data/a2c_returns.csv"
}

ppo_ablation_paths = {
    f"PPO Hidden {h}": f"data/ppo_hidden{h}_returns.csv"
    for h in [256]
}

# Combine all for iteration
all_paths = {**baseline_paths, **ppo_ablation_paths}

# Plot style
sns.set_palette("colorblind")
sns.set_style("whitegrid")

# Output directory
os.makedirs("plots", exist_ok=True)

# Plot config
DOWNSAMPLE_FACTOR = 250
SMOOTHING_WINDOW = 5000

# Plot
plt.figure(figsize=(12, 6))

for label, path in all_paths.items():
    if not os.path.exists(path):
        print(f"Skipping {label}: File not found at {path}")
        continue

    df = pd.read_csv(path)
    if "Total Steps" not in df.columns or "Episode Reward" not in df.columns:
        print(f"Skipping {label}: Missing required columns.")
        continue

    grouped = df.groupby("Total Steps")["Episode Reward"]
    mean_rewards = grouped.mean()
    std_rewards = grouped.std()

    steps = mean_rewards.index
    smoothed_means = mean_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    smoothed_stds = std_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    steps = steps[::DOWNSAMPLE_FACTOR]
    smoothed_means = smoothed_means[::DOWNSAMPLE_FACTOR]
    smoothed_stds = smoothed_stds[::DOWNSAMPLE_FACTOR]

    plt.plot(steps, smoothed_means, label=label)
    plt.fill_between(
        steps,
        smoothed_means - smoothed_stds,
        smoothed_means + smoothed_stds,
        alpha=0.2
    )

# Labels and formatting
plt.xlabel("Total Environment Steps", fontsize=12)
plt.ylabel("Episode Reward", fontsize=12)
plt.title("Comparison of Policy Gradient Methods and PPO Hidden Layer Ablations", fontsize=14)
plt.legend(fontsize=10, title="Algorithm", title_fontsize=12, loc="lower right", frameon=True)
plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
plt.minorticks_on()
plt.xlim(left=0)
plt.ylim(bottom=0)

# Save
plot_path = "plots/ppo_ablation_full_comparison.png"
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Full comparison plot saved to {plot_path}")
