import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

# Hyperparameters
ACTOR_LR = 0.001
CRITIC_LR = 0.001
EPISODES = 5000
GAMMA = 0.99
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class SharedActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_size=64):
        super().__init__()

        self.actor_shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        actor_features = self.actor_shared(x)
        action_probs = self.actor(actor_features)
        value = self.critic(x).squeeze(-1)
        return action_probs, value

# Instantiate model
model = SharedActorCritic(obs_dim, n_actions).to(device)
actor_params = list(model.actor.parameters()) + list(model.actor_shared.parameters())
critic_params = list(model.critic.parameters())

optimizer = optim.Adam([
    {'params': actor_params, 'lr': ACTOR_LR},
    {'params': critic_params, 'lr': CRITIC_LR}
])

# Training
episode_returns = []

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs, value = model(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
        done_tensor = torch.tensor(done, dtype=torch.float32).to(device)

        _, next_value = model(next_state_tensor)
        td_target = reward_tensor + GAMMA * next_value * (1 - done_tensor)
        td_error = td_target - value

        actor_loss = -dist.log_prob(action) * td_error.detach()
        critic_loss = td_error.pow(2)
        total_loss = actor_loss + critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    episode_returns.append(total_reward)
    print(f"Episode {episode+1}: Reward = {total_reward}")

env.close()

# Save
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/actor_critic_best_combo.csv", index=False)
print("Saved returns to data/actor_critic_best_combo.csv")
