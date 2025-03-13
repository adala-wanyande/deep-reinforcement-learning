import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# Experiment parameters
NUM_RUNS = 5  # Number of independent runs per gamma setting
MAX_ENV_STEPS = int(10e6)  # 10 million environment steps per run
LEARNING_RATE = 0.0005
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01
GAMMA_SETTINGS = {
    "Small Gamma (0.85)": 0.85,  # Focuses on short-term rewards
    "Medium Gamma (0.95)": 0.95,  # Balanced
    "High Gamma (0.99)": 0.99  # Focuses on long-term rewards
}

# Create the environment
env = gym.make("CartPole-v1")

# Define a single hidden layer neural network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # Single hidden layer with 64 neurons
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Naive DQN Agent (No Experience Replay, No Target Network)
class NaiveDQN:
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma  # Discount factor for future rewards
        self.q_network = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = EPSILON

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return torch.argmax(self.q_network(torch.tensor(state, dtype=torch.float32))).item()

    def train(self, state, action, reward, next_state, done):
        """Trains the Q-network using only the most recent experience."""
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Compute Q-values
        q_value = self.q_network(state).gather(0, action)

        # Compute TD target (bootstrapped target)
        next_q_value = self.q_network(next_state).max().detach()
        target = reward + self.gamma * next_q_value * (1 - done)  # Use different gamma values

        # Compute loss and update
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

# Training function
def train_dqn(gamma):
    """Trains the DQN agent with a specific gamma value and averages over NUM_RUNS."""
    all_results = []

    for run in range(NUM_RUNS):
        agent = NaiveDQN(env, gamma)
        run_results = []
        total_steps = 0  # Track total environment steps

        while total_steps < MAX_ENV_STEPS:
            state, _ = env.reset()
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < 200:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Train on only the most recent transition
                agent.train(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                step_count += 1
                total_steps += 1  # Update global step count

                # Stop if we've reached the total step limit
                if total_steps >= MAX_ENV_STEPS:
                    break

            agent.decay_epsilon()
            run_results.append(total_reward)

            print(f"Gamma={gamma}, Run {run+1}, Step {total_steps}: Episode {len(run_results)}, Reward = {total_reward}")

        all_results.append(run_results)

    # Pad all runs to the same length
    max_episodes = max(len(run) for run in all_results)
    for run in all_results:
        while len(run) < max_episodes:
            run.append(np.nan)  # Fill missing episodes with NaN for uniformity

    return np.array(all_results)  # Now has uniform shape

# Run experiments for each gamma setting
results_dict = {}
for setting, gamma in GAMMA_SETTINGS.items():
    results = train_dqn(gamma)
    mean_results = np.nanmean(results, axis=0)  # Average over runs
    std_results = np.nanstd(results, axis=0)  # Standard deviation
    results_dict[setting] = (mean_results, std_results)

# Create a DataFrame for CSV storage
csv_data = []
for setting, (mean_results, std_results) in results_dict.items():
    for episode, (mean, std) in enumerate(zip(mean_results, std_results), start=1):
        csv_data.append([setting, episode, mean, std])

df = pd.DataFrame(csv_data, columns=["Gamma Setting", "Episode", "Mean Total Reward", "Std Dev"])

# Ensure output directory exists
output_dir = os.getcwd()  # Saves in the current working directory
csv_path = os.path.join(output_dir, "ablation_gamma_results.csv")
plot_path = os.path.join(output_dir, "ablation_gamma_plot.png")

# Save results to CSV
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

# Plot results
plt.figure(figsize=(10, 5))
for setting, (mean_results, _) in results_dict.items():
    smoothed = pd.Series(mean_results).rolling(window=20, min_periods=1).mean()
    plt.plot(range(1, len(mean_results) + 1), smoothed, label=f"{setting}")

plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("CartPole Performance - Ablation Study on Discount Factor (Gamma)")
plt.legend()
plt.grid()

# Save plot
plt.savefig(plot_path)
plt.show()
print(f"Plot saved to {plot_path}")

env.close()
