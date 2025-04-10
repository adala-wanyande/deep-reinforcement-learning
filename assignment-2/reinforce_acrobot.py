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
NUM_RUNS = 5
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Env setup
env_name = "Acrobot-v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Return computation
def compute_returns(rewards, gamma):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)

# Training loop
all_data = []

for run in range(1, NUM_RUNS + 1):
    print(f"\n=== RUN {run} ===")
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = SharedActorCritic(obs_dim, n_actions, actor_hidden=128, critic_dims=[256, 128]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_env_steps = 0
    episode = 0

    while total_env_steps < TOTAL_STEPS:
        state, _ = env.reset(seed=SEED + run)
        done = False
        log_probs = []
        rewards = []
        total_reward = 0
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
            steps_this_episode += 1
            state = next_state

        # Update model
        returns = compute_returns(rewards, GAMMA)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_env_steps += steps_this_episode
        episode += 1
        all_data.append({
            "Run": run,
            "Episode": episode,
            "Total Steps": total_env_steps,
            "Episode Reward": total_reward
        })

        print(f"[Run {run}] Ep {episode}: Reward = {total_reward} | Total Steps = {total_env_steps}")

    env.close()

# Save combined data
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(all_data)
df.to_csv("data/reinforce_acrobot_results.csv", index=False)
print("Saved results to data/reinforce_acrobot_results.csv")
