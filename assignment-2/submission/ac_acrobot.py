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
TOTAL_STEPS = 1_000_000
NUM_RUNS = 5
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Environment settings
env_name = "Acrobot-v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Results holder
all_data = []

for run in range(1, NUM_RUNS + 1):
    print(f"\n=== RUN {run} ===")
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = SharedActorCritic(obs_dim, n_actions, actor_hidden=128, critic_dims=[256, 128]).to(device)
    actor_params = list(model.actor.parameters()) + list(model.actor_shared.parameters())
    critic_params = list(model.critic.parameters())
    optimizer = optim.Adam([
        {"params": actor_params, "lr": LEARNING_RATE},
        {"params": critic_params, "lr": LEARNING_RATE},
    ])

    total_env_steps = 0
    episode = 0

    while total_env_steps < TOTAL_STEPS:
        state, _ = env.reset(seed=SEED + run)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            action_probs, value = model(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, next_value = model(next_state_tensor)
                td_target = reward + GAMMA * next_value * (1 - float(done))

            td_error = td_target - value
            critic_loss = td_error.pow(2)
            actor_loss = -dist.log_prob(action) * td_error.detach()

            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward
            steps += 1

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
df.to_csv("data/actor_critic_acrobot_results.csv", index=False)
print("Saved results to data/actor_critic_acrobot_results.csv")
