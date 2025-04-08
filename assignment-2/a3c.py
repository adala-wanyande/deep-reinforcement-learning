# a3c.py
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from models import SharedActorCritic  # 👈 shared import

from collections import deque

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
EPISODES = 5000
NUM_ENVS = 4
N_STEPS = 5
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment
envs = [gym.make("CartPole-v1") for _ in range(NUM_ENVS)]
obs_dim = envs[0].observation_space.shape[0]
n_actions = envs[0].action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared model and optimizer
model = SharedActorCritic(obs_dim, n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

episode_returns = [0 for _ in range(NUM_ENVS)]
episode_counter = 0
reward_log = []

# Initialize environment states
states = [env.reset()[0] for env in envs]

while episode_counter < EPISODES:
    trajectories = [[] for _ in range(NUM_ENVS)]

    for step in range(N_STEPS):
        state_tensors = torch.tensor(states, dtype=torch.float32).to(device)
        with torch.no_grad():
            action_probs, _ = model(state_tensors)
        dists = torch.distributions.Categorical(action_probs)
        actions = dists.sample()

        for i in range(NUM_ENVS):
            next_state, reward, terminated, truncated, _ = envs[i].step(actions[i].item())
            done = terminated or truncated

            trajectories[i].append((states[i], actions[i].item(), reward, next_state, done))
            episode_returns[i] += reward
            states[i] = next_state

            if done:
                reward_log.append(episode_returns[i])
                episode_counter += 1
                print(f"Episode {episode_counter}: Return = {episode_returns[i]}")
                episode_returns[i] = 0
                states[i], _ = envs[i].reset()

    # Train on trajectories
    for env_traj in trajectories:
        if not env_traj:
            continue

        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*env_traj)
        states_tensor = torch.tensor(states_batch, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(actions_batch, dtype=torch.int64).to(device)
        rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
        next_states_tensor = torch.tensor(next_states_batch, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones_batch, dtype=torch.float32).to(device)

        _, next_values = model(next_states_tensor)
        returns = []
        R = next_values[-1] * (1 - dones_tensor[-1])
        for r, d in zip(reversed(rewards_tensor), reversed(dones_tensor)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)
        returns = torch.stack(returns).detach()

        action_probs, values = model(states_tensor)
        dists = torch.distributions.Categorical(action_probs)
        log_probs = dists.log_prob(actions_tensor)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Cleanup
for env in envs:
    env.close()

# Save
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, len(reward_log) + 1)), "Return": reward_log})
df.to_csv("data/a3c_returns.csv", index=False)
print("Saved returns to data/a3c_returns.csv")
