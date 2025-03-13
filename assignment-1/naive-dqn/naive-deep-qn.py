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
MAX_ENV_STEPS = 100000  # Total environment steps per run
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01

# Create the environment
env = gym.make("CartPole-v1")

# Define a minimal neural network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 24),  # Minimal network with small hidden layer
            nn.ReLU(),
            nn.Linear(24, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # Increase from 24 to 64 units
            nn.ReLU(),
            nn.Linear(64, 64),  # New hidden layer for better function approximation
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Output layer remains unchanged
        )

    def forward(self, x):
        return self.fc(x)


# Truly Naive DQN Agent (No Experience Replay, No Target Network)
class NaiveDQN:
    def __init__(self, env):
        self.env = env
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
        target = reward + GAMMA * next_q_value * (1 - done)

        # Compute loss and update
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

# Training function
def train_dqn():
    agent = NaiveDQN(env)
    results = []
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
        results.append([len(results) + 1, total_reward])

        print(f"Step {total_steps}: Episode {len(results)}, Reward = {total_reward}")

    return np.array(results)  # Shape: (num_episodes, 2) [Episode, Reward]

# Run single iteration of Naive DQN
results = train_dqn()

# Create Pandas DataFrame
df = pd.DataFrame(results, columns=["Episode", "Total Reward"])

# Ensure output directory exists
output_dir = os.getcwd()  # Saves in the current working directory
csv_path = os.path.join(output_dir, "naive_dqn_single_run_results.csv")
plot_path = os.path.join(output_dir, "naive_dqn_single_run_plot.png")

# Save results to CSV
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(df["Episode"], df["Total Reward"], label="Episodic Return", alpha=0.8, color="blue")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("CartPole Performance - Naive Deep Q-Learning (Single Run)")
plt.legend()
plt.grid()

# Save plot
plt.savefig(plot_path)
plt.show()
print(f"Plot saved to {plot_path}")

env.close()
