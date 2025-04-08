# reinforce_lunarlander.py
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from models import SharedActorCritic

# Hyperparameters
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPISODES = 500
SEED = 42

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment setup change for all files:
env = gym.make("Acrobot-v1")
obs_dim = env.observation_space.shape[0]  # 8 for LunarLander
n_actions = env.action_space.n  # 4 for LunarLander
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust model initialization (recommended sizes for LunarLander):
model = SharedActorCritic(obs_dim, n_actions, actor_hidden=128, critic_dims=[256, 128]).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Return computation
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

# Training loop
episode_returns = []

for episode in range(EPISODES):
    state, _ = env.reset(seed=SEED)
    done = False
    log_probs = []
    rewards = []
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs, _ = model(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        rewards.append(reward)
        total_reward += reward
        state = next_state

    returns = compute_returns(rewards, GAMMA)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -torch.sum(torch.stack(log_probs) * returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    episode_returns.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward}")

env.close()

# Save returns
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/reinforce_acrobot_returns.csv", index=False)
print("Saved returns to data/reinforce_acrobot_returns.csv")
