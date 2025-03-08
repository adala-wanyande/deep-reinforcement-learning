# No reinforcement learning here, just testing out the environment
import gymnasium
import numpy as np

# Create the environment with rendering enabled
env = gymnasium.make('CartPole-v1', render_mode="human")

# Number of episodes and timesteps per episode
num_episodes = 50
max_timesteps = 100

# Run multiple episodes
for episode in range(num_episodes):
    state, _ = env.reset()  # Reset environment and get initial state
    print(f"Episode {episode + 1}/{num_episodes} started")

    for t in range(max_timesteps):
        action = env.action_space.sample()  # Select a random action
        next_state, reward, terminated, truncated, info = env.step(action)
        
        print(f"Time step {t + 1}: Action {action}, Reward {reward}")

        if terminated or truncated:
            print(f"Episode {episode + 1} ended after {t + 1} timesteps")
            break

# Close the environment
env.close()
