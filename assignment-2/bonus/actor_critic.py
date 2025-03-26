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

# Actor network (Gaussian policy for continuous actions)
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.shared(x)
        mu = self.mu_head(x)
        std = self.log_std.exp()
        return mu, std

# Critic network (state-value function)
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)

# Initialize actor, critic, optimizers
actor = Actor().to(device)
critic = Critic().to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

episode_returns = []

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        mu, std = actor(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        action_clipped = action.clamp(act_low.to(device), act_high.to(device)).cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(action_clipped)
        done = terminated or truncated

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(device)

        value = critic(state_tensor)
        next_value = critic(next_state_tensor)
        td_target = reward_tensor + GAMMA * next_value * (1 - done_tensor)
        td_error = td_target - value

        # Critic update
        critic_loss = td_error.pow(2)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update
        actor_loss = -log_prob * td_error.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state
        total_reward += reward

    episode_returns.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

env.close()

# Save results
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/actor_critic_halfcheetah_returns.csv", index=False)
print("Saved returns to data/actor_critic_halfcheetah_returns.csv")