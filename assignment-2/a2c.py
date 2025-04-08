# a2c.py
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

# Initialize shared model and optimizer
model = SharedActorCritic(obs_dim, n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

episode_returns = []

for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    states, actions, rewards, dones = [], [], [], []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action_probs, _ = model(state_tensor)
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

        if len(states) >= N_STEPS or done:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, bootstrap_value = model(next_state_tensor)
                bootstrap = bootstrap_value * (1.0 - dones[-1])

            returns = []
            R = bootstrap
            for r, d in zip(reversed(rewards), reversed(dones)):
                R = r + GAMMA * R * (1 - d)
                returns.insert(0, R)

            states_tensor = torch.stack(states)
            actions_tensor = torch.stack(actions)
            returns_tensor = torch.stack(returns).detach()

            action_probs, values = model(states_tensor)
            dists = torch.distributions.Categorical(action_probs)
            log_probs = dists.log_prob(actions_tensor)

            advantages = returns_tensor - values

            # Actor update
            actor_loss = -(log_probs * advantages.detach()).mean()

            # Critic update
            critic_loss = advantages.pow(2).mean()

            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            states, actions, rewards, dones = [], [], [], []

    episode_returns.append(total_reward)
    print(f"Episode {episode+1}: Reward = {total_reward}")

env.close()

# Save returns
os.makedirs("data", exist_ok=True)
df = pd.DataFrame({"Episode": list(range(1, EPISODES + 1)), "Return": episode_returns})
df.to_csv("data/a2c_returns.csv", index=False)
print("Saved returns to data/a2c_returns.csv")
