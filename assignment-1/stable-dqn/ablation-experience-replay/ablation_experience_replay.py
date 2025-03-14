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
NUM_RUNS = 2  # Number of independent runs per configuration
MAX_ENV_STEPS = 100000  # 10 million environment steps per run
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01
MEMORY_SIZE = 50000  # Replay buffer size
BATCH_SIZE = 64  # Mini-batch size

# Create the environment
env = gym.make("CartPole-v1")

# Define a single hidden layer neural network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Agent with and without Experience Replay
class DQNAgent:
    def __init__(self, env, use_experience_replay):
        self.env = env
        self.use_experience_replay = use_experience_replay
        self.q_network = QNetwork(env.observation_space.shape[0], env.action_space.n)

        # Initialize experience replay buffer if using it
        if self.use_experience_replay:
            self.memory = deque(maxlen=MEMORY_SIZE)

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

    def store_experience(self, state, action, reward, next_state, done):
        """Stores transition in experience replay buffer if enabled."""
        if self.use_experience_replay:
            self.memory.append((state, action, reward, next_state, done))

    def train(self, state, action, reward, next_state, done):
        """Trains the Q-network using either direct updates or experience replay."""
        if self.use_experience_replay:
            if len(self.memory) < BATCH_SIZE:
                return  # Wait until buffer fills up
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
        else:
            # Standard DQN update
            state = torch.tensor(state, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            action = torch.tensor(action).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)

            q_value = self.q_network(state).gather(0, action)
            next_q_value = self.q_network(next_state).max().detach()
            target = reward + GAMMA * next_q_value * (1 - done)

            loss = self.criterion(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

def train_dqn(use_experience_replay):
    """Trains the DQN agent with and without experience replay, averaging over NUM_RUNS."""
    all_results = []

    for run in range(NUM_RUNS):
        agent = DQNAgent(env, use_experience_replay)
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

                # Store in replay buffer if enabled
                agent.store_experience(state, action, reward, next_state, done)

                # Train the network
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

            print(f"Experience Replay={use_experience_replay}, Run {run+1}, Step {total_steps}, Episode {len(run_results)}, Reward = {total_reward}")

        all_results.append(run_results)

    # **Fix: Pad all runs to the same length**
    max_episodes = max(len(run) for run in all_results)  # Find the longest run
    for run in all_results:
        while len(run) < max_episodes:
            run.append(np.nan)  # Fill missing episodes with NaN for uniform shape

    return np.array(all_results)  # Now has uniform shape

# Run experiments for both configurations
results_dict = {
    "Naive DQN (No Experience Replay)": train_dqn(use_experience_replay=False),
    "DQN with Experience Replay": train_dqn(use_experience_replay=True),
}

# Create a DataFrame for CSV storage
csv_data = []
for setting, results in results_dict.items():
    mean_results = np.nanmean(results, axis=0)  # Average over runs
    std_results = np.nanstd(results, axis=0)  # Standard deviation
    for episode, (mean, std) in enumerate(zip(mean_results, std_results), start=1):
        csv_data.append([setting, episode, mean, std])

df = pd.DataFrame(csv_data, columns=["Configuration", "Episode", "Mean Total Reward", "Std Dev"])

# Save results to CSV
df.to_csv("ablation_experience_replay_results.csv", index=False)

# Plot results with downsampling and smoothing
plt.figure(figsize=(10, 5))
for setting, results in results_dict.items():
    mean_results = np.nanmean(results, axis=0)
    std_results = np.nanstd(results, axis=0)

    # Apply rolling mean for smoothing
    smoothed_means = pd.Series(mean_results).rolling(window=50, min_periods=1).mean()
    smoothed_std_devs = pd.Series(std_results).rolling(window=50, min_periods=1).mean()

    # Downsample episodes to avoid excessive points
    downsample_factor = 100  # Only plot every 100th episode
    episodes = np.arange(1, len(mean_results) + 1)
    episodes_downsampled = episodes[::downsample_factor]
    smoothed_means_downsampled = smoothed_means[::downsample_factor]
    smoothed_std_devs_downsampled = smoothed_std_devs[::downsample_factor]

    plt.plot(episodes_downsampled, smoothed_means_downsampled, label=setting)
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
plt.savefig("ablation_experience_replay_plot.png")
plt.show()

env.close()
