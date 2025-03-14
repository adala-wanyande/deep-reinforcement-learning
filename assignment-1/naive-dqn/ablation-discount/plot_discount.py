import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import os
import numpy as np

# Load CSV file from the "data" folder
csv_path = "../data/ablation_gamma_results.csv"
df = pd.read_csv(csv_path)

# Ensure data is grouped by gamma setting
configurations = df["Gamma Setting"].unique()

# Apply seaborn's colorblind theme
sns.set_palette("colorblind")  
sns.set_style("whitegrid")  

# Ensure "visuals" directory exists
os.makedirs("../visuals", exist_ok=True)

# Downsampling factor (set to None for full resolution)
DOWNSAMPLE_FACTOR = 50  # Adjusted to prevent excessive skipping
SMOOTHING_WINDOW = 500  # Adjusted to provide a balanced smoothness

# Prepare the figure
plt.figure(figsize=(10, 5))

for config in configurations:
    df_config = df[df["Gamma Setting"] == config]

    # Group by steps to compute mean and std
    grouped = df_config.groupby("Total Steps")["Episode Reward"]
    mean_rewards = grouped.mean()
    std_rewards = grouped.std()

    steps = mean_rewards.index
    smoothed_means = mean_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    smoothed_stds = std_rewards.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # Downsampling
    if DOWNSAMPLE_FACTOR:
        steps = steps[::DOWNSAMPLE_FACTOR]
        smoothed_means = smoothed_means[::DOWNSAMPLE_FACTOR]
        smoothed_stds = smoothed_stds[::DOWNSAMPLE_FACTOR]

    plt.plot(steps, smoothed_means, label=f"Gamma = {config}")
    plt.fill_between(
        steps,
        smoothed_means - smoothed_stds,
        smoothed_means + smoothed_stds,
        alpha=0.2
    )

# Force the axes to start at (0,0)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.xlabel("Total Steps", fontsize=12)
plt.ylabel("Episode Reward", fontsize=12)
plt.title("Ablation Study on Discount Factor (Steps vs Reward)", fontsize=14)

# Improve the legend
plt.legend(title="Discount Factor (Gamma)", fontsize=10, title_fontsize=12, loc="upper left", frameon=True)

# **Adding Enhanced Grid Lines**
plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.7)  # Major grid
plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)  # Minor grid
plt.minorticks_on()  # Enable minor ticks

# Save and show the plot
plot_path = "../visuals/ablation_gamma_plot.png"
plt.savefig(plot_path, dpi=300)  # High-resolution save
plt.show()

print(f"Plot saved to {plot_path}")
