import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# File paths
reinforce_path = "data/reinforce_acrobot_results.csv"
ac_path = "data/actor_critic_acrobot_results.csv"
a2c_path = "data/a2c_acrobot_results.csv"

# Load data
df_reinforce = pd.read_csv(reinforce_path)
df_ac = pd.read_csv(ac_path)
df_a2c = pd.read_csv(a2c_path)

# Store for iteration
experiments = {
    "REINFORCE": df_reinforce,
    "Actor-Critic (AC)": df_ac,
    "Advantage Actor-Critic (A2C)": df_a2c,
}

# Plot style
sns.set_palette("colorblind")
sns.set_style("whitegrid")

# Create output directory
os.makedirs("plots", exist_ok=True)

# Plot config
DOWNSAMPLE_FACTOR = 250
SMOOTHING_WINDOW = 5000  # For larger-scale smoothing with step-based x-axis

# Plot
plt.figure(figsize=(10, 5))

for label, df in experiments.items():
    if "Total Steps" not in df.columns or "Episode Reward" not in df.columns:
        print(f"Skipping {label}: missing necessary columns")
        continue

    grouped = df.groupby("Total Steps")["Episode Reward"]
    mean_rewards = grouped.mean()
    std_rewards = grouped.std()

    steps = mean_rewards.index
    smoothed_means = mean_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    smoothed_stds = std_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # Downsample
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
plt.xlim(left=0)
plt.ylim(bottom=-510)
plt.xlabel("Total Environment Steps", fontsize=12)
plt.ylabel("Episode Reward", fontsize=12)
plt.title("Comparison of Policy Gradient Methods on Acrobot", fontsize=14)

plt.legend(title="Algorithm", fontsize=10, title_fontsize=12, loc="lower right", frameon=True)
plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
plt.minorticks_on()

# Save and display
plot_path = "plots/policy_gradient_comparison_acrobot.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Plot saved to {plot_path}")
