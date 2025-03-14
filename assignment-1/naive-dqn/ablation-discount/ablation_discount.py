import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import pandas as pd
import os
import random

# Experiment parameters
NUM_RUNS = 5  # Number of independent runs per gamma setting
MAX_ENV_STEPS = int(1e6)  # 1 million environment steps per run
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
            nn.Linear(input_dim, 64),  
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Naive DQN Agent (No Experience Replay, No Target Network)
class NaiveDQN:
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma  
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
        target = reward + self.gamma * next_q_value * (1 - done)

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
    """Trains the DQN agent with a specific gamma value and logs steps vs episode rewards."""
    all_results = []

    for run in range(NUM_RUNS):
        agent = NaiveDQN(env, gamma)
        run_results = []
        total_steps = 0  

        while total_steps < MAX_ENV_STEPS:
            state, _ = env.reset()
            done = False
            episode_reward = 0  

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Train on the most recent transition
                agent.train(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                total_steps += 1  

                # Stop if max steps are reached
                if total_steps >= MAX_ENV_STEPS:
                    break

            # Log **steps vs episode rewards**
            run_results.append((total_steps, episode_reward))
            
            # Decay epsilon after each episode
            agent.decay_epsilon()

            print(f"Gamma={gamma}, Run {run+1}, Steps={total_steps}, Episode Reward={episode_reward}")

            # Stop if max steps are reached
            if total_steps >= MAX_ENV_STEPS:
                break

        all_results.append(run_results)

    return all_results  

# Run experiments for each gamma setting
results_dict = {}
for setting, gamma in GAMMA_SETTINGS.items():
    results = train_dqn(gamma)
    results_dict[setting] = results

# Create a DataFrame for CSV storage
csv_data = []
for setting, results in results_dict.items():
    for run_results in results:
        for step, episode_reward in run_results:
            csv_data.append([setting, step, episode_reward])

df = pd.DataFrame(csv_data, columns=["Gamma Setting", "Total Steps", "Episode Reward"])

# Ensure the "data" directory exists
os.makedirs("../data", exist_ok=True)

# Save results to CSV inside "data" folder
csv_path = os.path.join("../data", "ablation_gamma_results.csv")
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

env.close()
