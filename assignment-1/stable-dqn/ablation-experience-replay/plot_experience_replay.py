import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
csv_path = "ablation_experience_replay_results.csv"
df = pd.read_csv(csv_path)

# Ensure data is grouped by configuration
configurations = df["Configuration"].unique()

# Plot results
plt.figure(figsize=(10, 5))

for config in configurations:
    df_config = df[df["Configuration"] == config]

    episodes = df_config["Episode"].values
    mean_rewards = df_config["Mean Total Reward"].values
    std_devs = df_config["Std Dev"].values

    # Apply rolling mean for smoothing
    smoothed_means = pd.Series(mean_rewards).rolling(window=50, min_periods=1).mean()  # Smoother moving average
    smoothed_std_devs = pd.Series(std_devs).rolling(window=50, min_periods=1).mean()  # Smooth std deviation

    # Downsample episodes to avoid excessive points
    downsample_factor = 100  # Only plot every 5th episode
    episodes_downsampled = episodes[::downsample_factor]
    smoothed_means_downsampled = smoothed_means[::downsample_factor]
    smoothed_std_devs_downsampled = smoothed_std_devs[::downsample_factor]

    plt.plot(episodes_downsampled, smoothed_means_downsampled, label=config)
    plt.fill_between(
        episodes_downsampled, 
        smoothed_means_downsampled - smoothed_std_devs_downsampled, 
        smoothed_means_downsampled + smoothed_std_devs_downsampled, 
        alpha=0.2
    )

plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Ablation Study: Impact of Experience Replay")
plt.legend()
plt.grid()

# Save and show the plot
plot_path = "ablation_experience_replay_plot.png"
plt.savefig(plot_path)
plt.show()

print(f"Plot saved to {plot_path}")
