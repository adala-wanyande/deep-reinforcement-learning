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
NUM_ENVS = 4
N_STEPS = 5

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Environment setup
envs = [gym.make("HalfCheetah-v4") for _ in range(NUM_ENVS)]
obs_dim = envs[0].observation_space.shape[0]
act_dim = envs[0].action_space.shape[0]
act_low = torch.tensor(envs[0].action_space.low, dtype=torch.float32)
act_high = torch.tensor(envs[0].action_space.high, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-Critic shared network
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.mu(x), self.log_std.exp(), self.value(x).squeeze(-1)

model = ActorCritic().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

states = [env.reset()[0] for env in envs]
episode_returns = [0 for _ in range(NUM_ENVS)]
reward_log = []
episode_counter = 0

while episode_counter < EPISODES:
    trajectories = [[] for _ in range(NUM_ENVS)]

    for step in range(N_STEPS):
        state_tensors = torch.tensor(states, dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, std, _ = model(state_tensors)
        dist = torch.distributions.Normal(mu, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=1)
        actions_clipped = actions.clamp(act_low.to(device), act_high.to(device)).cpu().numpy()

        for i in range(NUM_ENVS):
            next_state, reward, terminated, truncated, _ = envs[i].step(actions_clipped[i])
            done = terminated or truncated
            trajectories[i].append((states[i], actions[i], reward, next_state, done, log_probs[i]))
            episode_returns[i] += reward
            states[i] = next_state

            if done:
                reward_log.append(episode_returns[i])
                episode_counter += 1
                print(f"Episode {episode_counter}: Return = {episode_returns[i]:.2f}")
                episode_returns[i] = 0
                states[i], _ = envs[i].reset()

    # Training
    for traj in trajectories:
        if len(traj) == 0:
            continue

        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, log_probs_batch = zip(*traj)
        states_tensor = torch.tensor(states_batch, dtype=torch.float32).to(device)
        actions_tensor = torch.stack(actions_batch).to(device)
        rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
        next_states_tensor = torch.tensor(next_states_batch, dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(dones_batch, dtype=torch.float32).to(device)
        log_probs_tensor = torch.stack(log_probs_batch).to(device)

        with torch.no_grad():
            _, _, next_values = model(next_states_tensor)
            R = next_values[-1] * (1.0 - dones_tensor[-1])

        returns = []
        for r, d in zip(reversed(rewards_tensor), reversed(dones_tensor)):
            R = r + GAMMA * R * (1 - d)
            returns.insert(0, R)
        returns_tensor = torch.stack(returns).detach()

        _, _, values_tensor = model(states_tensor)
        advantages = returns_tensor - values_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs_tensor * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Close envs
for env in envs:
    env.close()

# Save
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, len(reward_log)+1)), "Return": reward_log})
df.to_csv("data/a3c_halfcheetah_returns.csv", index=False)
print("Saved returns to data/a3c_halfcheetah_returns.csv")  