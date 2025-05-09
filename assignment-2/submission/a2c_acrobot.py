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
TOTAL_STEPS = 1_000_000
NUM_RUNS = 5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

env_name = "Acrobot-v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_data = []

for run in range(1, NUM_RUNS + 1):
    print(f"\n=== RUN {run} ===")
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = SharedActorCritic(obs_dim, n_actions, actor_hidden=128, critic_dims=[256, 128]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_env_steps = 0
    episode = 0

    while total_env_steps < TOTAL_STEPS:
        state, _ = env.reset(seed=SEED + run)
        done = False
        total_reward = 0
        steps = 0

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
            steps += 1

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

        total_env_steps += steps
        episode += 1
        all_data.append({
            "Run": run,
            "Episode": episode,
            "Total Steps": total_env_steps,
            "Episode Reward": total_reward
        })

        print(f"[Run {run}] Episode {episode}: Reward = {total_reward:.2f} | Steps = {total_env_steps}")

    env.close()

# Save results
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(all_data)
df.to_csv("data/a2c_acrobot_results.csv", index=False)
print("Saved results to data/a2c_acrobot_results.csv")
