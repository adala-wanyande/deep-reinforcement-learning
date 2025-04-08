# actor_critic_lunarlander.py
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
EPISODES = 500
SEED = 42

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment setup change for all files:
env = gym.make("Acrobot-v1")
obs_dim = env.observation_space.shape[0]  # 8 for LunarLander
n_actions = env.action_space.n  # 4 for LunarLander
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust model initialization (recommended sizes for LunarLander):
model = SharedActorCritic(obs_dim, n_actions, actor_hidden=128, critic_dims=[256, 128]).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        # Critic loss
        critic_loss = td_error.pow(2)

        # Actor loss
        actor_loss = -dist.log_prob(action) * td_error.detach()

        # Joint update
        total_loss = actor_loss + critic_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    episode_returns.append(total_reward)
    print(f"Episode {episode+1}: Reward = {total_reward:.2f}")

env.close()

# Save returns
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/actor_critic_acrobot_returns.csv", index=False)
print("Saved returns to data/actor_critic_acrobot_returns.csv")
