# Assignment 1: Deep Q-Learning on CartPole

## Introduction
This assignment explores reinforcement learning (RL) by implementing Deep Q-Learning (DQN) on the CartPole-v1 environment from Gymnasium. The project follows a structured approach, starting from environment setup, testing random action selection, and progressing towards function approximation using neural networks with experience replay and target networks.

## Step 1: Setting Up the Environment
The first step was to install the necessary dependencies and test whether the Gymnasium CartPole environment was working correctly.

### Installation
To set up the environment, install the required libraries listed in `requirements.txt`:

```sh
pip install -r requirements.txt
```

### Testing CartPole with Random Actions
Before implementing any learning, we tested random action selection to verify that the environment works as expected.

```python
import gymnasium as gym
import numpy as np

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode="human")

num_episodes = 5
max_timesteps = 100

for episode in range(num_episodes):
    state, _ = env.reset()
    print(f"Episode {episode + 1}/{num_episodes} started")

    for t in range(max_timesteps):
        action = env.action_space.sample()  # Take a random action
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Time step {t + 1}: Action {action}, Reward {reward}")

        if terminated or truncated:
            print(f"Episode {episode + 1} ended after {t + 1} timesteps")
            break

env.close()
```

### Observations
- The agent took random actions without any learning.
- Performance did not improve over episodes.
- The pole fell quickly most of the time.
- Since no past experiences were stored, no reinforcement learning occurred.

This step confirmed that our environment setup was correct, and we could move on to implementing an actual RL agent.

