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
TOTAL_STEPS = 1_000_000
RUNS = 5
SEED = 42

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment setup
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compute discounted returns
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

# Run experiments
all_results = []

for run in range(1, RUNS + 1):
    print(f"\n=== Run {run} ===")
    model = SharedActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    episode = 0
    steps = 0

    while steps < TOTAL_STEPS:
        state, _ = env.reset(seed=SEED + run + episode)  # Different seed for diversity
        done = False
        total_reward = 0
        rewards = []
        log_probs = []
        steps_this_episode = 0

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
            steps_this_episode += 1

        steps += steps_this_episode
        episode += 1

        returns = compute_returns(rewards, GAMMA)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -torch.sum(torch.stack(log_probs) * returns)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_results.append({
            "Run": run,
            "Episode": episode,
            "Total Steps": steps,
            "Episode Reward": total_reward
        })

        print(f"Run {run} | Episode {episode} | Reward = {total_reward:.1f} | Total Steps = {steps}")

env.close()

# Save CSV
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv("data/reinforce_returns.csv", index=False)
print("Saved returns to data/reinforce_returns.csv")