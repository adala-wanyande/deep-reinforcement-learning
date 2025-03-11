import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Experiment parameters
NUM_RUNS = 5  # Number of independent runs for averaging
NUM_EPISODES = 1000  # Number of episodes per run (same as Tabular Q-Learning)
MAX_TIMESTEPS = 200  # Max steps per episode

# Create the environment
env = gym.make("CartPole-v1")

def run_random_policy():
    """Runs a random policy and returns episodic rewards over multiple runs."""
    all_results = []

    for run in range(NUM_RUNS):
        run_results = []

        for episode in range(NUM_EPISODES):
            state, _ = env.reset()
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < MAX_TIMESTEPS:
                action = env.action_space.sample()  # Take a random action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step_count += 1

            run_results.append(total_reward)

        all_results.append(run_results)

    return np.array(all_results)  # Shape: (NUM_RUNS, NUM_EPISODES)

# Run multiple iterations of the random policy
results = run_random_policy()
mean_results = np.mean(results, axis=0)  # Average over runs
std_results = np.std(results, axis=0)  # Standard deviation

# Create Pandas DataFrame
df = pd.DataFrame({
    "Episode": np.arange(1, NUM_EPISODES + 1),
    "Mean Total Reward": mean_results,
    "Std Dev": std_results
})

# Compute rolling average for smooth plotting
df["Smoothed Mean Return"] = df["Mean Total Reward"].rolling(window=20, min_periods=1).mean()

# Ensure output directory exists
output_dir = os.getcwd()  # Saves in the current working directory
csv_path = os.path.join(output_dir, "random_actions_results.csv")
plot_path = os.path.join(output_dir, "random_actions_plot.png")

# Save results to CSV
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

# Plot results with error bars
plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Mean Total Reward"], label="Mean Episodic Return", alpha=0.5, color="blue")
plt.plot(df["Episode"], df["Smoothed Mean Return"], label="Smoothed Mean (Rolling Avg)", linewidth=2, color="red")
plt.fill_between(df["Episode"], df["Mean Total Reward"] - df["Std Dev"], df["Mean Total Reward"] + df["Std Dev"], color="blue", alpha=0.2)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("CartPole Performance - Random Actions (Averaged)")
plt.legend()
plt.grid()

# Save plot
plt.savefig(plot_path)
plt.show()
print(f"Plot saved to {plot_path}")

env.close()
