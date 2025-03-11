import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from collections import deque

# Experiment parameters
NUM_RUNS = 5  # Number of independent runs for averaging
NUM_EPISODES = 1000  # Number of episodes per run
MAX_TIMESTEPS = 200  # Max steps per episode
LEARNING_RATE = 0.01
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 32
MEMORY_SIZE = 10000

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

# Naive DQN Agent (No Experience Replay, No Target Network)
class NaiveDQN:
    def __init__(self, env):
        self.env = env
        self.q_network = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return torch.argmax(self.q_network(torch.tensor(state, dtype=torch.float32))).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Stores transition in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Trains the Q-network using mini-batch updates."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions)).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)


        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions)

        # Compute TD targets
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        targets = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute loss and update
        loss = self.criterion(q_values.squeeze(), targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

# Training function
def train_dqn():
    all_results = []

    for run in range(NUM_RUNS):
        agent = NaiveDQN(env)
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

                agent.store_experience(state, action, reward, next_state, done)
                agent.train()
                state = next_state
                total_reward += reward
                step_count += 1

            agent.decay_epsilon()
            run_results.append(total_reward)

        all_results.append(run_results)

    return np.array(all_results)  # Shape: (NUM_RUNS, NUM_EPISODES)

# Run multiple iterations of DQN
results = train_dqn()
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
csv_path = os.path.join(output_dir, "naive_dqn_results.csv")
plot_path = os.path.join(output_dir, "naive_dqn_plot.png")

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
plt.title("CartPole Performance - Naive Deep Q-Learning (DQN)")
plt.legend()
plt.grid()

# Save plot
plt.savefig(plot_path)
plt.show()
print(f"Plot saved to {plot_path}")

env.close()
