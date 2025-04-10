import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from models import SharedActorCritic

# Constants
LEARNING_RATE = 0.0005
GAMMA = 0.99
SEED = 42
REPEATS = 5
TOTAL_STEPS = 1_000_000
ENV_NAME = "CartPole-v1"

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Output
os.makedirs("data", exist_ok=True)
all_runs = []

for run in range(1, REPEATS + 1):
    print(f"\n--- Starting Run {run} ---")

    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SharedActorCritic(obs_dim, n_actions).to(device)
    actor_params = list(model.actor.parameters()) + list(model.actor_shared.parameters())
    critic_params = list(model.critic.parameters())

    optimizer = optim.Adam([
        {'params': actor_params, 'lr': LEARNING_RATE},
        {'params': critic_params, 'lr': LEARNING_RATE}
    ])

    episode_returns = []
    total_steps_list = []
    total_env_steps = 0

    while total_env_steps < TOTAL_STEPS:
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0

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
            actor_loss = -dist.log_prob(action) * td_error.detach()
            critic_loss = td_error.pow(2)
            total_loss = actor_loss + critic_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward
            episode_steps += 1

        total_env_steps += episode_steps
        episode_returns.append(total_reward)
        total_steps_list.append(total_env_steps)

        print(f"Run {run}, Episode {len(episode_returns)}: Reward = {total_reward}, Steps = {total_env_steps}")

    # Store run data
    run_df = pd.DataFrame({
        "Run": run,
        "Episode": range(1, len(episode_returns) + 1),
        "Total Steps": total_steps_list,
        "Episode Reward": episode_returns
    })
    all_runs.append(run_df)
    env.close()

# Save combined CSV
final_df = pd.concat(all_runs, ignore_index=True)
final_df.to_csv("data/actor_critic_returns.csv", index=False)
print("Saved results to data/actor_critic_returns.csv")
