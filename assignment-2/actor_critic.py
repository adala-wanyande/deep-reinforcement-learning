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

# Environment setup
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor network
class Actor(nn.Module):
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

# Critic network
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)

# Initialize networks and optimizers
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
        action_probs = actor(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(device)

        # Compute value estimates
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
        actor_loss = -dist.log_prob(action) * td_error.detach()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state
        total_reward += reward

    episode_returns.append(total_reward)
    print(f"Episode {episode+1}: Reward = {total_reward}")

env.close()

# Save returns
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/actor_critic_returns.csv", index=False)
print("Saved returns to data/actor_critic_returns.csv")