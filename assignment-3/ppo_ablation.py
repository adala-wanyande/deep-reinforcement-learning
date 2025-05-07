import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from collections import deque
import torch.nn as nn
from torch.distributions import Categorical
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--hidden", type=int, default=64, help="Size of hidden layers")
args = parser.parse_args()
HIDDEN_SIZE = args.hidden

# PPO hyperparameters
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPS = 0.2
EPOCHS = 10
BATCH_SIZE = 64
STEPS_PER_UPDATE = 2048
TOTAL_TIMESTEPS = 1_000_000
SEED = 42
REPEATS = 1  # Set to 1 for single run
ENV_NAME = "CartPole-v1"



# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared).squeeze(-1)


# Rollout buffer
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

def compute_returns_and_advantages(rewards, dones, values, gamma=GAMMA, lam=0.95):
    returns, advs = [], []
    gae = 0
    next_value = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advs.insert(0, gae)
        returns.insert(0, gae + values[i])
        next_value = values[i]
    return torch.tensor(returns), torch.tensor(advs)

def ppo_update(model, optimizer, buffer):
    states = torch.stack(buffer.states)
    actions = torch.tensor(buffer.actions)
    old_log_probs = torch.tensor(buffer.log_probs)
    returns, advantages = compute_returns_and_advantages(
        buffer.rewards, buffer.dones, buffer.values
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(EPOCHS):
        idxs = np.arange(len(states))
        np.random.shuffle(idxs)
        for i in range(0, len(states), BATCH_SIZE):
            batch = idxs[i:i+BATCH_SIZE]
            logits, value = model(states[batch])
            dist = Categorical(logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions[batch])

            ratio = torch.exp(new_log_probs - old_log_probs[batch])
            surr1 = ratio * advantages[batch]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[batch]
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (returns[batch] - value).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Output
os.makedirs("data", exist_ok=True)
all_runs = []

for run in range(1, REPEATS + 1):
    print(f"\n--- Starting Run {run} ---")

    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    buffer = RolloutBuffer()

    episode_returns = []
    total_steps_list = []
    total_env_steps = 0
    episode_count = 0
    current_episode_reward = 0

    obs, _ = env.reset()

    while total_env_steps < TOTAL_TIMESTEPS:
        for _ in range(STEPS_PER_UPDATE):
            state_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits, value = model(state_tensor)
            dist = Categorical(logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, done, truncated, _ = env.step(action.item())
            buffer.states.append(state_tensor.cpu().numpy())
            buffer.actions.append(action.item())
            buffer.log_probs.append(log_prob.item())
            buffer.rewards.append(reward)
            buffer.dones.append(done or truncated)
            buffer.values.append(value.item())

            current_episode_reward += reward
            total_env_steps += 1
            obs = next_obs

            if done or truncated:
                episode_returns.append(current_episode_reward)
                total_steps_list.append(total_env_steps)
                print(f"Run {run}, Episode {len(episode_returns)}: Reward = {current_episode_reward}, Steps = {total_env_steps}")
                current_episode_reward = 0
                episode_count += 1
                obs, _ = env.reset()

        # Convert buffer data to tensors on the correct device before updating
        buffer.states = [torch.tensor(s, dtype=torch.float32).to(device) for s in buffer.states]
        buffer.log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)

        ppo_update(model, optimizer, buffer)
        buffer.clear()

    # Store run data
    run_df = pd.DataFrame({
        "Run": run,
        "Episode": range(1, len(episode_returns) + 1),
        "Total Steps": total_steps_list,
        "Episode Reward": episode_returns
    })
    all_runs.append(run_df)
    env.close()

# Save combined CSV
final_df = pd.concat(all_runs, ignore_index=True)
final_df.to_csv(f"data/ppo_hidden{HIDDEN_SIZE}_returns.csv", index=False)
print(f"Saved results to data/ppo_hidden{HIDDEN_SIZE}_returns.csv")
