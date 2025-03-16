import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import os
import pandas as pd
import numpy as np
import random
from collections import deque

# Hyperparameters
NUM_RUNS = 5               # Number of independent runs
MAX_ENV_STEPS = int(1e6)   # 1 million environment steps per run
GAMMA = 0.99               # High discount factor (long-term rewards)
EPSILON = 0.99             # Low exploration (high exploitation)
EPSILON_DECAY = 0.999      # Decay rate
MIN_EPSILON = 0.01         # Minimum epsilon
LEARNING_RATE = 0.001      # High learning rate
BATCH_SIZE = 64            # Mini-batch size for experience replay
BUFFER_SIZE = 10000        # Experience replay buffer size
TARGET_UPDATE_FREQ = 500  # Frequency for updating target network

# Define Neural Network (Medium-sized: 64-64-2)
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# DQN Agent with Experience Replay & Target Network
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.q_network = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.target_network = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Sync initial weights
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.epsilon = EPSILON
        self.memory = deque(maxlen=BUFFER_SIZE)  # Experience Replay Buffer

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Ensure batch dimension
                return torch.argmax(self.q_network(state_tensor)).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Stores experiences in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Trains the Q-network using experience replay."""
        if len(self.memory) < BATCH_SIZE:
            return  # Don't train until buffer has enough experiences

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values for current state-action pairs
        q_values = self.q_network(states).gather(1, actions).squeeze(1)

        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        # Compute loss and update
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Sync target network with main Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay epsilon for exploration-exploitation balance."""
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

# Training function
def train_dqn(run_id):
    """Trains the DQN agent for a single run and logs steps vs rewards."""
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)

    total_steps = 0
    run_results = []  # Stores results for this run

    while total_steps < MAX_ENV_STEPS:
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_experience(state, action, reward, next_state, done)  # Store experience
            agent.train()  # Train using replay buffer

            state = next_state
            episode_reward += reward
            total_steps += 1

            if total_steps % TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            # Stop if max steps are reached
            if total_steps >= MAX_ENV_STEPS:
                break

        run_results.append((run_id, total_steps, episode_reward))  # Log run ID, total steps, and reward
        agent.decay_epsilon()

        print(f"Run {run_id}: Steps={total_steps}, Episode Reward={episode_reward}")

        if total_steps >= MAX_ENV_STEPS:
            break

    env.close()
    return run_results  # Store step-based rewards for this run

# Run Training for multiple runs
all_results = []
for run in range(1, NUM_RUNS + 1):
    run_results = train_dqn(run)
    all_results.extend(run_results)

# Save results to CSV
df = pd.DataFrame(all_results, columns=["Run ID", "Total Steps", "Episode Reward"])
os.makedirs("../data", exist_ok=True)
csv_path = "../data/stable_dqn_both_results.csv"
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")
