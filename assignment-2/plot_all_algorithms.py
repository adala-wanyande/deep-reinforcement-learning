import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# File paths
paths = {
    "REINFORCE": "data/reinforce_returns.csv",
    "Actor-Critic (AC)": "data/actor_critic_returns.csv",
    "Advantage Actor-Critic (A2C)": "data/a2c_returns.csv",
    "Asynchronous Actor-Critic (A3C)": "data/a3c_returns.csv"
}

# Load and preprocess data
experiments = {}
for label, path in paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Align to unified naming convention
        if "Episode" not in df.columns or "Return" not in df.columns:
            continue
        df_grouped = df.groupby("Episode")["Return"]
        mean_rewards = df_grouped.mean()
        std_rewards = df_grouped.std()
        experiments[label] = (mean_rewards, std_rewards)
    else:
        print(f"Warning: File not found for {label}: {path}")

# Visualization settings
sns.set_palette("colorblind")
sns.set_style("whitegrid")
os.makedirs("plots", exist_ok=True)

# Plot config
DOWNSAMPLE_FACTOR = 20
SMOOTHING_WINDOW = 1000

plt.figure(figsize=(10, 6))

for label, (mean_rewards, std_rewards) in experiments.items():
    episodes = mean_rewards.index
    smoothed_means = mean_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    smoothed_stds = std_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # Downsample
    if DOWNSAMPLE_FACTOR:
        episodes = episodes[::DOWNSAMPLE_FACTOR]
        smoothed_means = smoothed_means[::DOWNSAMPLE_FACTOR]
        smoothed_stds = smoothed_stds[::DOWNSAMPLE_FACTOR]

    plt.plot(episodes, smoothed_means, label=label)
    plt.fill_between(episodes,
                     smoothed_means - smoothed_stds,
                     smoothed_means + smoothed_stds,
                     alpha=0.2)

# Axis settings
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel("Episodes", fontsize=12)
plt.ylabel("Episode Reward", fontsize=12)
plt.title("Policy Gradient Algorithms on CartPole", fontsize=14)

# Legend
plt.legend(title="Algorithm", fontsize=10, title_fontsize=12, loc="upper left", frameon=True)

# Grid lines
plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
plt.minorticks_on()

# Save
plot_path = "plots/policy_gradient_comparison.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Plot saved to {plot_path}")