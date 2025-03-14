import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import pandas as pd
import os
import random

# Experiment parameters
NUM_RUNS = 5  # Number of independent runs per learning rate
MAX_ENV_STEPS = int(1e6)  # 1 million environment steps per run
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.01
LEARNING_RATES = [0.0001, 0.0005, 0.001]  # Small, Medium, High learning rates

# Create the environment
env = gym.make("CartPole-v1")

# Define a minimal neural network for Q-learning
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
    def __init__(self, env, learning_rate):
        self.env = env
        self.q_network = QNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
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
def train_dqn(learning_rate):
    """Trains the DQN agent with a specific learning rate while capping total steps globally."""
    all_results = []

    for run in range(NUM_RUNS):
        total_steps = 0  # ✅ Reset step count per run
        agent = NaiveDQN(env, learning_rate)
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

            print(f"LR={learning_rate}, Run={run+1}, Steps={total_steps}, Episode Reward={episode_reward}")

            # Stop if max steps are reached
            if total_steps >= MAX_ENV_STEPS:
                break

        all_results.append(run_results)

    return all_results  # Store step-based rewards

# Run experiments for each learning rate
results_dict = {}
for lr in LEARNING_RATES:
    results = train_dqn(lr)
    results_dict[lr] = results

# Create a DataFrame for CSV storage
csv_data = []
for lr, results in results_dict.items():
    for run_results in results:
        for step, avg_reward in run_results:
            csv_data.append([lr, step, avg_reward])

df = pd.DataFrame(csv_data, columns=["Learning Rate", "Total Steps", "Average Reward"])

# Ensure the "data" directory exists
os.makedirs("../data", exist_ok=True)

# Save results to CSV
csv_path = os.path.join("../data", "ablation_learning_rate_results.csv")
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

env.close()
