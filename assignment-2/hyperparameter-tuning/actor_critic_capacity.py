# actor_critic_capacity.py
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import argparse
import ast
from models import SharedActorCritic

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument('--actor-lr', type=float, default=1e-3)
parser.add_argument('--critic-lr', type=float, default=1e-3)
parser.add_argument('--episodes', type=int, default=1000)
parser.add_argument('--critic-dims', type=str, default='[128,128]', help="Critic architecture like '[256,128]'")
parser.add_argument('--save-path', type=str, default=None, help="Path to save results CSV")
args = parser.parse_args()

# Hyperparameters
ACTOR_LR = args.actor_lr
CRITIC_LR = args.critic_lr
EPISODES = args.episodes
CRITIC_DIMS = ast.literal_eval(args.critic_dims)
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
model = SharedActorCritic(obs_dim, n_actions, critic_dims=CRITIC_DIMS).to(device)
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

        actor_loss = -dist.log_prob(action) * td_error.detach()
        critic_loss = td_error.pow(2)
        total_loss = actor_loss + critic_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

    episode_returns.append(total_reward)
    print(f"Episode {episode + 1}: Reward = {total_reward}")

env.close()

# Save
os.makedirs("data", exist_ok=True)

if args.save_path is not None:
    save_path = args.save_path
else:
    dims_str = "_".join(map(str, CRITIC_DIMS))
    save_path = f"data/actor_critic_dims_{dims_str}.csv"

df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv(save_path, index=False)
print(f"Saved returns to {save_path}")
