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
N_STEPS = 5
RUNS = 5
TOTAL_STEPS = 1_000_000
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment setup
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Results
all_results = []

for run in range(1, RUNS + 1):
    print(f"\n=== Run {run} ===")
    model = SharedActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    episode = 0
    total_env_steps = 0

    while total_env_steps < TOTAL_STEPS:
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps_this_episode = 0

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
            steps_this_episode += 1

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

                actor_loss = -(log_probs * advantages.detach()).mean()
                critic_loss = advantages.pow(2).mean()

                loss = actor_loss + critic_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                states, actions, rewards, dones = [], [], [], []

        episode += 1
        total_env_steps += steps_this_episode
        all_results.append({
            "Run": run,
            "Episode": episode,
            "Total Steps": total_env_steps,
            "Episode Reward": total_reward
        })
        print(f"Run {run} | Ep {episode} | Steps {total_env_steps} | Reward {total_reward:.1f}")

env.close()

# Save results
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(all_results)
df.to_csv("data/a2c_returns.csv", index=False)
print("Saved returns to data/a2c_returns.csv")
