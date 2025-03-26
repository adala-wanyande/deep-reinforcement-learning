import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Hyperparameters
LEARNING_RATE = 0.0003
GAMMA = 0.99
EPISODES = 1000
SEED = 42

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment
env = gym.make("HalfCheetah-v4")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_low = torch.tensor(env.action_space.low, dtype=torch.float32)
act_high = torch.tensor(env.action_space.high, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy Network for continuous actions (Gaussian)
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.shared(state)
        mu = self.mu_head(x)
        std = self.log_std.exp()
        return mu, std

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

# Initialize policy and optimizer
policy = GaussianPolicy(obs_dim, act_dim).to(device)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

episode_returns = []

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    log_probs = []
    rewards = []
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        mu, std = policy(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        action_clipped = action.clamp(act_low.to(device), act_high.to(device)).detach().cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action_clipped)
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        state = next_state

    returns = compute_returns(rewards, GAMMA)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

    loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns.to(device))]).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_returns.append(total_reward)
    print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

env.close()

# Save results
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/reinforce_halfcheetah_returns.csv", index=False)
print("Saved returns to data/reinforce_halfcheetah_returns.csv")