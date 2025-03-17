import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import os
import numpy as np

# File paths
naive_dqn_path = "./naive-dqn/data/naive_dqn_best_hyperparams_results.csv"
experience_replay_path = "./stable-dqn/data/stable_dqn_experience_replay_results.csv"
target_network_path = "./stable-dqn/data/stable_dqn_target_results.csv"
both_stabilizations_path = "./stable-dqn/data/stable_dqn_both_results.csv"

# Load data
df_naive = pd.read_csv(naive_dqn_path)
df_experience_replay = pd.read_csv(experience_replay_path)
df_target_network = pd.read_csv(target_network_path)
df_both_stabilizations = pd.read_csv(both_stabilizations_path)

# Dictionary to store datasets for easy iteration
experiments = {
    "Naive DQN (No Stabilization)": df_naive,
    "Experience Replay Only": df_experience_replay,
    "Target Network Only": df_target_network,
    "Experience Replay + Target Network": df_both_stabilizations,
}

# Apply seaborn's colorblind-friendly palette
sns.set_palette("colorblind")
sns.set_style("whitegrid")

# Ensure visuals directory exists
os.makedirs("../visuals", exist_ok=True)

# Downsampling factor
DOWNSAMPLE_FACTOR = 20  # Adjust to control the number of points plotted
SMOOTHING_WINDOW = 10000  # Adjust smoothing effect

# Prepare the figure
plt.figure(figsize=(10, 5))

for label, df in experiments.items():
    # Ensure data is grouped correctly
    grouped = df.groupby("Total Steps")["Episode Reward"]
    mean_rewards = grouped.mean()
    std_rewards = grouped.std()

    steps = mean_rewards.index
    smoothed_means = mean_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    smoothed_stds = std_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # Downsample for readability
    if DOWNSAMPLE_FACTOR:
        steps = steps[::DOWNSAMPLE_FACTOR]
        smoothed_means = smoothed_means[::DOWNSAMPLE_FACTOR]
        smoothed_stds = smoothed_stds[::DOWNSAMPLE_FACTOR]

    # Plot with error boundaries
    plt.plot(steps, smoothed_means, label=label)
    plt.fill_between(
        steps,
        smoothed_means - smoothed_stds,
        smoothed_means + smoothed_stds,
        alpha=0.2
    )

# Configure plot aesthetics
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel("Total Steps", fontsize=12)
plt.ylabel("Episode Reward", fontsize=12)
plt.title("Comparison of Naive DQN vs. Stabilized DQN Variants", fontsize=14)

# Improve the legend
plt.legend(title="Training Configuration", fontsize=10, title_fontsize=12, loc="upper left", frameon=True)

# Add enhanced grid lines
plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)
plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
plt.minorticks_on()

# Save and display the plot
plot_path = "../visuals/dqn_comparison_plot.png"
plt.savefig(plot_path, dpi=300)  # High-resolution save
plt.show()

print(f"Plot saved to {plot_path}")
