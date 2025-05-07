import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyperparameter settings
learning_rate = 0.001
gamma = 0.99
num_envs = 5
max_steps = 1000000
fine_interval = 1000
coarse_interval = 10000
hidden_layer = 256
num_runs = 5
update_every = 128
ppo_epochs = 10
clip_epsilon = 0.2
value_coef = 0.5
entropy_coef = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Create the CartPole environment
def make_env():
    return gym.make("CartPole-v1")

# PPO Network
class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(PPONetwork, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_size, action_dim),
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        policy_logits = self.policy(shared_features)
        value = self.value(shared_features)
        return policy_logits, value

def compute_gae(rewards, values, next_value, masks, gamma=0.99, lambda_=0.95):
    returns = []
    gae = 0
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[step + 1]
        
        delta = rewards[step] + gamma * next_val * masks[step] - values[step]
        gae = delta + gamma * lambda_ * masks[step] * gae
        returns.insert(0, gae + values[step])
    
    return returns

# PPO training loop
def train_ppo(env):
    all_rewards = []

    for run in range(num_runs):
        model = PPONetwork(env.single_observation_space.shape[0],
                          env.single_action_space.n,
                          hidden_layer).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device)
        episode_rewards = np.zeros(num_envs)
        total_rewards = []
        rewards_per_step = []
        step_count = 0

        print(f"Starting PPO Run {run+1}/{num_runs}")
        rewards_per_step.append(0.0)
        print(f"PPO Run {run+1}, Step {step_count}/{max_steps}, Mean Reward: 0.0")

        while step_count < max_steps:
            # Collect trajectories
            states_batch = []
            actions_batch = []
            old_log_probs_batch = []
            rewards_batch = []
            values_batch = []
            masks_batch = []

            for _ in range(update_every):
                # Get action from policy
                with torch.no_grad():
                    logits, value = model(state)
                    probs = torch.softmax(logits, dim=1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    old_log_prob = dist.log_prob(action)

                # Execute action in environment
                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                step_count += num_envs

                # Store trajectory information
                states_batch.append(state)
                actions_batch.append(action)
                old_log_probs_batch.append(old_log_prob)
                rewards_batch.append(torch.FloatTensor(reward).to(device))
                values_batch.append(value.squeeze())
                masks_batch.append(torch.FloatTensor(1 - done).to(device))

                episode_rewards += reward
                state = torch.FloatTensor(next_state).to(device)

                # Handle episode termination
                for i in range(num_envs):
                    if done[i]:
                        total_rewards.append(episode_rewards[i])
                        episode_rewards[i] = 0
                        s, _ = env.envs[i].reset()
                        state[i] = torch.FloatTensor(s).to(device)

                # Log progress
                if step_count <= 10000 and step_count % fine_interval == 0:
                    mean_reward = np.mean(total_rewards[-100:]) if total_rewards else 0.0
                    print(f"PPO Run {run+1}, Step {step_count}/{max_steps}, Mean Reward: {mean_reward:.2f}")
                    rewards_per_step.append(mean_reward)
                elif step_count > 10000 and step_count % coarse_interval == 0:
                    mean_reward = np.mean(total_rewards[-100:]) if total_rewards else 0.0
                    print(f"PPO Run {run+1}, Step {step_count}/{max_steps}, Mean Reward: {mean_reward:.2f}")
                    rewards_per_step.append(mean_reward)

            # Compute advantages and returns
            with torch.no_grad():
                _, next_value = model(state)
                next_value = next_value.squeeze()

            values_batch = torch.stack(values_batch)
            returns = compute_gae(rewards_batch, values_batch, next_value, masks_batch)
            returns = torch.stack(returns)
            advantages = returns - values_batch

            # Convert batches to tensors
            states_batch = torch.stack(states_batch)
            actions_batch = torch.stack(actions_batch)
            old_log_probs_batch = torch.stack(old_log_probs_batch)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            for _ in range(ppo_epochs):
                # Get current policy and value predictions
                logits, values = model(states_batch)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                curr_log_probs = dist.log_prob(actions_batch)
                entropy = dist.entropy().mean()

                # Compute ratio and clipped objective
                ratio = torch.exp(curr_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                
                # Compute losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((returns - values.squeeze()) ** 2).mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                # Update network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        all_rewards.append(rewards_per_step)
        print(f"PPO Run {run+1} completed. Final Mean Reward: {rewards_per_step[-1]:.2f}")

    return all_rewards

# Plotting results
def plot_results(rewards_runs):
    fine_steps = np.arange(0, 10001, fine_interval) / 1000
    coarse_steps = np.arange(20000, max_steps + 1, coarse_interval) / 1000
    steps = np.concatenate([fine_steps, coarse_steps])
    window_size = 10

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    max_len = max(len(run) for run in rewards_runs)
    padded_runs = np.array([np.pad(run, (0, max_len - len(run)), 'edge') for run in rewards_runs])
    mean_rewards = np.mean(padded_runs, axis=0)
    std_rewards = np.std(padded_runs, axis=0)

    rewards_smooth = np.convolve(mean_rewards, np.ones(window_size)/window_size, mode='valid')
    padding = np.linspace(mean_rewards[0], rewards_smooth[0], window_size - 1)
    rewards_smooth = np.concatenate([padding, rewards_smooth])
    steps_smooth = steps[:len(rewards_smooth)]

    std_rewards = std_rewards[window_size-1:]
    std_padding = np.linspace(0, std_rewards[0], window_size - 1)
    std_rewards = np.concatenate([std_padding, std_rewards])

    plt.plot(steps_smooth, rewards_smooth, label="PPO", marker='o', linestyle='-', linewidth=2)
    plt.fill_between(steps_smooth, rewards_smooth - std_rewards, rewards_smooth + std_rewards, alpha=0.2)

    plt.xlabel('Steps (x1000)')
    plt.ylabel('Mean Reward')
    plt.title('PPO on CartPole-v1')
    plt.legend()
    plt.grid(True)

    plt.ylim(0, 600)
    plt.xlim(0, max_steps / 1000)
    plt.xticks(np.arange(0, max_steps / 1000 + 1, 100))
    plt.yticks(np.arange(100, 601, 100))

    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)

    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.yaxis.set_label_coords(-0.05, 0.5)

    plt.savefig("ppo.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    env = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    rewards = train_ppo(env)
    plot_results(rewards)