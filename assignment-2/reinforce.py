import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
EPISODES = 5000
SEED = 42

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

# Training Loop
policy = PolicyNetwork().to(device)
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
        probs = policy(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        rewards.append(reward)
        total_reward += reward
        state = next_state

    returns = compute_returns(rewards, GAMMA)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_returns.append(total_reward)
    print(f"Episode {episode+1}: Reward = {total_reward}")

env.close()

# Save results
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/reinforce_returns.csv", index=False)
print("Saved returns to data/reinforce_returns.csv")