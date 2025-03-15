import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import os
import numpy as np

# Load CSV file
csv_path = "../data/stable_dqn_both_results.csv"
df = pd.read_csv(csv_path)

# Ensure "visuals" directory exists
os.makedirs("../visuals", exist_ok=True)

# Apply seaborn's colorblind theme
sns.set_palette("colorblind")  
sns.set_style("whitegrid")  

# Define downsampling and smoothing parameters
DOWNSAMPLE_FACTOR = 200  # Adjusted for clarity
SMOOTHING_WINDOW = 10000  # Ensures balanced smoothing

# Prepare the figure
plt.figure(figsize=(10, 5))

# Group data by "Total Steps"
grouped = df.groupby("Total Steps")

# Compute mean and standard deviation of "Episode Reward" at each step
mean_rewards = grouped["Episode Reward"].mean()
std_rewards = grouped["Episode Reward"].std()

steps = mean_rewards.index

# Apply smoothing using rolling mean
smoothed_means = mean_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
smoothed_stds = std_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

# Downsampling
if DOWNSAMPLE_FACTOR:
    steps = steps[::DOWNSAMPLE_FACTOR]
    smoothed_means = smoothed_means[::DOWNSAMPLE_FACTOR]
    smoothed_stds = smoothed_stds[::DOWNSAMPLE_FACTOR]

# Plot mean rewards
plt.plot(steps, smoothed_means, label="Mean Episode Reward", linewidth=2)

# Add error boundary (Standard Deviation)
plt.fill_between(
    steps,
    smoothed_means - smoothed_stds,
    smoothed_means + smoothed_stds,
    alpha=0.3,
    label="Standard Deviation"
)

# Force the axes to start at (0,0)
plt.xlim(left=0)
plt.ylim(bottom=0)

# Labels and Title
plt.xlabel("Total Steps", fontsize=12)
plt.ylabel("Episode Reward", fontsize=12)
plt.title("Stable DQN Performance (Steps vs Reward)", fontsize=14)

# Improve the legend
plt.legend(title="Performance Metrics", fontsize=10, title_fontsize=12, loc="upper left", frameon=True)

# **Adding Enhanced Grid Lines**
plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)  # Major grid
plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)  # Minor grid
plt.minorticks_on()  # Enable minor ticks

# Save and show the plot
plot_path = "../visuals/stable_dqn_both_plot.png"
plt.savefig(plot_path, dpi=300)  # High-resolution save
plt.show()

print(f"Plot saved to {plot_path}")
