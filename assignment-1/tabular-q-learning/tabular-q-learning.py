import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict

# Discretization settings
NUM_BINS = 10  # Number of bins per state dimension
NUM_RUNS = 5   # Number of independent runs for averaging
NUM_EPISODES = 1000  # Number of episodes per run
MAX_TIMESTEPS = 200  # Maximum steps per episode

# Create the environment
env = gym.make("CartPole-v1")

# Define discretization function
def discretize_state(state, bins):
    """Convert continuous state into discrete bins."""
    state_idx = tuple(np.digitize(s, bins[i]) for i, s in enumerate(state))
    return state_idx

# Tabular Q-Learning Agent
class TabularQLearning:
    def __init__(self, env, learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.env = env
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.state_bins = [np.linspace(-x, x, NUM_BINS) for x in env.observation_space.high]
        self.Q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        discrete_state = discretize_state(state, self.state_bins)
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[discrete_state])

    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule."""
        discrete_state = discretize_state(state, self.state_bins)
        next_discrete_state = discretize_state(next_state, self.state_bins)

        best_next_action = np.argmax(self.Q_table[next_discrete_state])
        td_target = reward + (1 - done) * self.gamma * self.Q_table[next_discrete_state][best_next_action]
        td_error = td_target - self.Q_table[discrete_state][action]
        self.Q_table[discrete_state][action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Training function
def train_q_learning():
    all_results = []  # Stores results from all runs

    for run in range(NUM_RUNS):
        agent = TabularQLearning(env)
        run_results = []

        for episode in range(NUM_EPISODES):
            state, _ = env.reset()
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < MAX_TIMESTEPS:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_count += 1

            agent.decay_epsilon()
            run_results.append(total_reward)

        all_results.append(run_results)

    return np.array(all_results)  # Shape: (NUM_RUNS, NUM_EPISODES)

# Run multiple training iterations and average results
results = train_q_learning()
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
csv_path = os.path.join(output_dir, "tabular_q_learning_results.csv")
plot_path = os.path.join(output_dir, "tabular_q_learning_plot.png")

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
plt.title("Tabular Q-Learning Performance on CartPole (Averaged)")
plt.legend()
plt.grid()

# Save plot
plt.savefig(plot_path)
plt.show()
print(f"Plot saved to {plot_path}")

env.close()
