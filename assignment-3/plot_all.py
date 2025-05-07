import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
import numpy as np

# --- Configuration ---
# List of hidden layer sizes to process
hidden_sizes = [64, 128, 256, 512]
# File template for input CSV files
file_template = "data/ppo_hidden{}_returns.csv"
# Factor by which to downsample the data for plotting (reduces number of points)
DOWNSAMPLE_FACTOR = 100 # Reduced for potentially smoother curves with more points
# Window size for rolling mean smoothing
SMOOTHING_WINDOW = 200 # Reduced for less aggressive smoothing, adjust as needed

# --- Plot Styling ---
# Use a seaborn style for a good base. 'seaborn-v0_8-whitegrid' is clean.
plt.style.use('seaborn-v0_8-whitegrid')
# Use a colorblind-friendly palette
sns.set_palette("colorblind")

# Customize matplotlib parameters for a more polished look
plt.rcParams.update({
    'font.family': 'sans-serif', # Consistent font family
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], # Preferred sans-serif fonts
    'axes.labelsize': 14,  # Font size for x and y labels
    'axes.titlesize': 16,  # Font size for the plot title
    'xtick.labelsize': 12, # Font size for x-axis tick labels
    'ytick.labelsize': 12, # Font size for y-axis tick labels
    'legend.fontsize': 11, # Font size for legend text
    'legend.title_fontsize': 13, # Font size for legend title
    'lines.linewidth': 2.0, # Thicker lines for mean rewards
    'lines.markersize': 5,   # Marker size if markers were used
    'figure.dpi': 100,       # Default DPI for the figure
    'savefig.dpi': 300,      # Higher DPI for saved figures
    'axes.edgecolor': 'gray', # Color of plot edges
    'grid.color': 'lightgray', # Color of grid lines
    'grid.linestyle': '--',    # Style of grid lines
    'grid.linewidth': 0.7,     # Width of grid lines
})

# Output directory for plots
os.makedirs("plots", exist_ok=True)

# --- Plot Generation ---
plt.figure(figsize=(12, 7)) # Adjusted figure size for better readability

# Define a list of distinct linestyles if needed, though color is primary differentiator
# linestyles = ['-', '--', '-.', ':']

for i, hidden in enumerate(hidden_sizes):
    filepath = file_template.format(hidden)
    if not os.path.exists(filepath):
        print(f"Skipping hidden size {hidden}: File '{filepath}' not found.")
        continue

    try:
        df = pd.read_csv(filepath)
        if "Total Steps" not in df.columns or "Episode Reward" not in df.columns:
            print(f"Skipping hidden size {hidden}: CSV missing 'Total Steps' or 'Episode Reward' column.")
            continue
    except pd.errors.EmptyDataError:
        print(f"Skipping hidden size {hidden}: File '{filepath}' is empty.")
        continue
    except Exception as e:
        print(f"Skipping hidden size {hidden}: Error reading file '{filepath}': {e}")
        continue

    # Group by 'Total Steps' and calculate mean and std of 'Episode Reward'
    # This handles cases where multiple episodes might end at the same total step count
    # within a single seed's run, or if data from multiple seeds were concatenated
    # without an explicit seed identifier.
    grouped = df.groupby("Total Steps")["Episode Reward"]
    mean_rewards = grouped.mean()
    std_rewards = grouped.std().fillna(0) # Fill NaN std (e.g. if only one data point for a step)

    # Apply smoothing
    # Ensure the window is not larger than the data
    current_smoothing_window = min(SMOOTHING_WINDOW, len(mean_rewards))
    if current_smoothing_window < 1: # Handle very short data
        current_smoothing_window = 1
        
    steps_raw = mean_rewards.index
    smoothed_means = mean_rewards.rolling(window=current_smoothing_window, min_periods=1, center=True).mean()
    # Smooth the standard deviation as well for a smoother error band
    smoothed_stds = std_rewards.rolling(window=current_smoothing_window, min_periods=1, center=True).mean()

    # Downsample after smoothing
    steps_plot = steps_raw[::DOWNSAMPLE_FACTOR]
    smoothed_means_plot = smoothed_means[::DOWNSAMPLE_FACTOR]
    smoothed_stds_plot = smoothed_stds[::DOWNSAMPLE_FACTOR]
    
    # Ensure no NaN values are passed to fill_between after downsampling
    smoothed_means_plot = smoothed_means_plot.fillna(method='ffill').fillna(method='bfill')
    smoothed_stds_plot = smoothed_stds_plot.fillna(method='ffill').fillna(method='bfill')


    plt.plot(steps_plot, smoothed_means_plot, label=f"{hidden} units") # More descriptive label
    plt.fill_between(
        steps_plot,
        (smoothed_means_plot - smoothed_stds_plot).clip(lower=0), # Clip at 0 for rewards
        (smoothed_means_plot + smoothed_stds_plot),
        alpha=0.15 # Slightly reduced alpha for better clarity if bands overlap
    )

# --- Final Plot Touches ---
# Add a horizontal line for CartPole solved criteria or max score if applicable
# CartPole-v1 is considered solved at an average reward of 475 over 100 consecutive trials.
# Max score is 500.
plt.axhline(y=500, color='grey', linestyle=':', linewidth=1.5, label='Max Score (500)')
plt.axhline(y=475, color='darkgrey', linestyle=':', linewidth=1.5, label='Solved Threshold (475)')


# Labels, Title, and Legend
plt.xlabel("Total Environment Steps")
plt.ylabel("Average Episode Reward")
plt.title("PPO Performance: Ablation Study on Hidden Layer Size (CartPole-v1)", fontweight='bold')

# Improve legend
legend = plt.legend(
    title="Network Hidden Layer Size",
    loc="lower right", # Consider 'best' or manual placement if overlap
    frameon=True,      # Add a frame to the legend
    shadow=False,       # Add a shadow effect
    fancybox=True,     # Use rounded corners for the legend box
    # bbox_to_anchor=(1.05, 1) # Example to place legend outside plot
)
legend.get_frame().set_edgecolor('dimgray')


# Axis limits and ticks
plt.xlim(left=0)
# Dynamically set x-axis limit based on max steps if desired, or keep fixed
# max_steps_data = df["Total Steps"].max() # Assuming last df is representative
# plt.xlim(0, max_steps_data * 1.05) # Add 5% padding
plt.ylim(0, 550) # Extend y-limit slightly above max possible score for CartPole

# Format x-axis ticks to be more readable (e.g., "100k" instead of "100000")
def k_formatter(x, pos):
    return f'{int(x/1000)}k' if x >= 1000 else int(x)
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(k_formatter))

# Add minor ticks for better visual guidance
plt.minorticks_on()
plt.grid(True, which='major', linestyle='-', linewidth='0.6', alpha=0.7)
plt.grid(True, which='minor', linestyle=':', linewidth='0.4', alpha=0.5)


# Ensure everything fits without overlapping
plt.tight_layout(pad=1.5) # Add some padding

# --- Save and Show ---
plot_path = "plots/ppo_hidden_ablation_enhanced.png"
plt.savefig(plot_path) # DPI is set in rcParams
plt.show()

print(f"Enhanced ablation plot saved to {plot_path}")