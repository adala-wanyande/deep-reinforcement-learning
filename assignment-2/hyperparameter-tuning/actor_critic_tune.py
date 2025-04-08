# actor_critic_tune.py
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import argparse
from models import SharedActorCritic

# CLI Args
parser = argparse.ArgumentParser()
parser.add_argument('--actor-lr', type=float, default=1e-3)
parser.add_argument('--critic-lr', type=float, default=1e-3)
parser.add_argument('--episodes', type=int, default=1000)
args = parser.parse_args()

ACTOR_LR = args.actor_lr
CRITIC_LR = args.critic_lr
EPISODES = args.episodes
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

# Model
model = SharedActorCritic(obs_dim, n_actions).to(device)

# Separate parameter groups
actor_params = list(model.actor.parameters())
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

        # Losses
        critic_loss = td_error.pow(2)
        actor_loss = -dist.log_prob(action) * td_error.detach()
        total_loss = actor_loss + critic_loss

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    episode_returns.append(total_reward)
    print(f"Episode {episode+1}: Reward = {total_reward}")

env.close()

# Save CSV
os.makedirs("data", exist_ok=True)
filename = f"data/actor_critic_actorlr_{ACTOR_LR:.0e}_criticlr_{CRITIC_LR:.0e}.csv"
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv(filename, index=False)
print(f"Saved returns to {filename}")
