import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import pandas as pd
import os
import random

# Experiment parameters
NUM_RUNS = 5  # Number of independent runs per network size
MAX_ENV_STEPS = int(1e6)  # 1 million environment steps per run
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01
NETWORK_SIZES = {
    "Small (32-1)": [32],  # Single hidden layer
    "Medium (64-64-2)": [64, 64],  # Two hidden layers
    "Large (128-128-128-3)": [128, 128, 128]  # Three hidden layers
}

# Create the environment
env = gym.make("CartPole-v1")

# Define different neural network architectures for Q-learning
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))  # Output layer
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

# Naive DQN Agent (No Experience Replay, No Target Network)
class NaiveDQN:
    def __init__(self, env, hidden_layers):
        self.env = env
        self.q_network = QNetwork(env.observation_space.shape[0], env.action_space.n, hidden_layers)
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
def train_dqn(hidden_layers):
    """Trains the DQN agent with a specific network architecture while capping total steps globally."""
    all_results = []

    for run in range(NUM_RUNS):
        total_steps = 0  # ✅ Reset step count per run
        agent = NaiveDQN(env, hidden_layers)
        run_results = []

        while total_steps < MAX_ENV_STEPS:
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Train on only the most recent transition
                agent.train(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                total_steps += 1  # ✅ Now tracks correctly within a run

                # Stop if max steps are reached
                if total_steps >= MAX_ENV_STEPS:
                    break

            # Log the reward **after each episode**
            run_results.append((total_steps, episode_reward))
            
            # Decay epsilon after each episode
            agent.decay_epsilon()

            print(f"Network={hidden_layers}, Run={run+1}, Steps={total_steps}, Episode Reward={episode_reward}")

            # Stop if max steps are reached
            if total_steps >= MAX_ENV_STEPS:
                break

        all_results.append(run_results)

    return all_results  # Store step-based rewards

# Run experiments for each network size
results_dict = {}
for setting, hidden_layers in NETWORK_SIZES.items():
    results = train_dqn(hidden_layers)
    results_dict[setting] = results

# Create a DataFrame for CSV storage
csv_data = []
for setting, results in results_dict.items():
    for run_results in results:
        for step, avg_reward in run_results:
            csv_data.append([setting, step, avg_reward])

df = pd.DataFrame(csv_data, columns=["Network Size", "Total Steps", "Average Reward"])

# Ensure the "data" directory exists
os.makedirs("../data", exist_ok=True)

# Save results to CSV
csv_path = os.path.join("../data", "ablation_network_size_results.csv")
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

env.close()
