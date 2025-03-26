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
N_STEPS = 5
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment
env = gym.make("HalfCheetah-v4")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_low = torch.tensor(env.action_space.low, dtype=torch.float32)
act_high = torch.tensor(env.action_space.high, dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor network
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

# Critic network
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

# Initialize
actor = Actor().to(device)
critic = Critic().to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

episode_returns = []

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    states, actions, rewards, dones, log_probs = [], [], [], [], []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        mu, std = actor(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        action_clipped = action.clamp(act_low.to(device), act_high.to(device)).cpu().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_clipped)
        done = terminated or truncated

        states.append(state_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
        dones.append(torch.tensor(done, dtype=torch.float32).to(device))

        state = next_state
        total_reward += reward

        if len(states) >= N_STEPS or done:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
            with torch.no_grad():
                bootstrap = critic(next_state_tensor) * (1.0 - dones[-1])

            returns = []
            R = bootstrap
            for r, d in zip(reversed(rewards), reversed(dones)):
                R = r + GAMMA * R * (1 - d)
                returns.insert(0, R)

            states_tensor = torch.stack(states)
            actions_tensor = torch.stack(actions)
            returns_tensor = torch.stack(returns).detach()
            values_tensor = critic(states_tensor)

            advantages = returns_tensor - values_tensor
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Actor update
            log_probs_tensor = torch.stack(log_probs)
            actor_loss = -(log_probs_tensor * advantages.detach()).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Critic update
            critic_loss = advantages.pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Clear buffers
            states, actions, rewards, dones, log_probs = [], [], [], [], []

    episode_returns.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

env.close()

# Save results
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/a2c_halfcheetah_returns.csv", index=False)
print("Saved returns to data/a2c_halfcheetah_returns.csv")