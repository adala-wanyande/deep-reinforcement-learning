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
N_STEPS = 5
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-Critic networks
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

# Initialize models and optimizers
actor = Actor().to(device)
critic = Critic().to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

episode_returns = []

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    states, actions, rewards, dones = [], [], [], []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs = actor(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state_tensor)
        actions.append(torch.tensor(action, dtype=torch.int64).to(device))
        rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
        dones.append(torch.tensor(done, dtype=torch.float32).to(device))

        state = next_state
        total_reward += reward

        # Update every N_STEPS or at episode end
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

            # Actor update
            logits = actor(states_tensor)
            dists = torch.distributions.Categorical(logits)
            log_probs = dists.log_prob(actions_tensor)
            actor_loss = -(log_probs * advantages.detach()).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Critic update
            critic_loss = advantages.pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            states, actions, rewards, dones = [], [], [], []

    episode_returns.append(total_reward)
    print(f"Episode {episode+1}: Reward = {total_reward}")

env.close()

# Save returns
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/a2c_returns.csv", index=False)
print("Saved returns to data/a2c_returns.csv")