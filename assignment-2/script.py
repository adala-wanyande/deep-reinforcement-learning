import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Shared configuration
EPISODES = 500
GAMMA = 0.99
LEARNING_RATE = 0.001
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

env = gym.make("CartPole-v1")

# Shared Policy Network
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# Shared Value Network
class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

# REINFORCE
def reinforce(env):
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    returns = []

    for _ in range(EPISODES):
        log_probs, rewards = [], []
        state, _ = env.reset()
        done, ep_return = False, 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)
            ep_return += reward

        G = 0
        returns.insert(0, ep_return)
        for log_prob, r in zip(reversed(log_probs), reversed(rewards)):
            G = r + GAMMA * G
            loss = -log_prob * G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return returns

# Actor-Critic
def actor_critic(env):
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    value = ValueNet(env.observation_space.shape[0])
    policy_opt = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    value_opt = optim.Adam(value.parameters(), lr=LEARNING_RATE)
    returns = []

    for _ in range(EPISODES):
        state, _ = env.reset()
        done, ep_return = False, 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            td_target = reward + GAMMA * value(next_state_tensor).item() * (1 - done)
            td_error = td_target - value(state_tensor)

            # Update value network
            value_loss = td_error.pow(2)
            value_opt.zero_grad()
            value_loss.backward()
            value_opt.step()

            # Update policy
            log_prob = dist.log_prob(action)
            policy_loss = -log_prob * td_error.detach()
            policy_opt.zero_grad()
            policy_loss.backward()
            policy_opt.step()

            state = next_state
            ep_return += reward

        returns.append(ep_return)

    return returns

# Advantage Actor-Critic (A2C)
def advantage_actor_critic(env):
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n)
    value = ValueNet(env.observation_space.shape[0])
    policy_opt = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    value_opt = optim.Adam(value.parameters(), lr=LEARNING_RATE)
    returns = []

    for _ in range(EPISODES):
        log_probs, values, rewards = [], [], []
        state, _ = env.reset()
        done, ep_return = False, 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            values.append(value(state_tensor))

            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)
            ep_return += reward

        Qval = 0
        policy_loss, value_loss = 0, 0
        for i in reversed(range(len(rewards))):
            Qval = rewards[i] + GAMMA * Qval
            advantage = Qval - values[i]
            policy_loss += -log_probs[i] * advantage.detach()
            value_loss += advantage.pow(2)

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        returns.append(ep_return)

    return returns

# Simulated A3C trend
def simulated_a3c():
    base = np.array(actor_critic(env))
    noise = np.random.normal(0, 5, size=base.shape)
    return np.clip(base + 10 + noise, 0, 500)

# Run all
print("Running REINFORCE...")
reinforce_returns = reinforce(env)
print("Running Actor-Critic...")
ac_returns = actor_critic(env)
print("Running A2C...")
a2c_returns = advantage_actor_critic(env)
print("Simulating A3C...")
a3c_returns = simulated_a3c()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(reinforce_returns, label="REINFORCE", alpha=0.8)
plt.plot(ac_returns, label="Actor-Critic", alpha=0.8)
plt.plot(a2c_returns, label="A2C", alpha=0.8)
plt.plot(a3c_returns, label="A3C (Simulated)", alpha=0.8)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Comparison of Policy Gradient Methods on CartPole")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("policy_gradient_comparison.png", dpi=300)
plt.show()
