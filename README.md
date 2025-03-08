# Deep Reinforcement Learning - MSc Course

This repository contains all my work for the Deep Reinforcement Learning (DRL) course this semester. It includes implementations, documentation, reports, and related resources for each assignment.

## Repository Structure

The repository is organized into folders, each corresponding to an assignment. Every folder contains:
- Source Code: Implementations of reinforcement learning algorithms.
- Reports: PDF write-ups following the course guidelines.
- Documentation: Additional insights, notes, or related references.
- README.md: A brief overview of the assignment.

### Assignments Overview

| Assignment | Description |
|------------|------------|
| A1: Q-Learning - Tabular & Deep | Implement tabular Q-learning and transition to Deep Q-Networks (DQN) using function approximation. Explore experience replay and target networks. |
| A2: Policy Gradient Methods | Implement and compare policy-based methods such as REINFORCE and Actor-Critic algorithms. Analyze convergence and policy stability. |
| A3: Advanced Deep RL Techniques | Explore deep reinforcement learning methods like PPO, DDPG, or SAC. Experiment with continuous control tasks. |
| A4: Final Project | Design and implement a custom RL agent for a chosen environment, applying techniques from previous assignments. |

## Setup Instructions

To run the code in this repository, follow these steps:

### Clone the Repository

```sh
git clone https://github.com/yourusername/deep-rl-course.git
cd deep-rl-course
```

### Create and Activate a Virtual Environment

```sh
python -m venv drl-env
source drl-env/bin/activate   # On Windows, use: drl-env\Scripts\activate
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Run an Assignment Script (Example: Assignment 1)

```sh
cd A1_Q_Learning
python train_q_learning.py
```

## Resources

- [OpenAI Gymnasium](https://gymnasium.farama.org/)
- [Deep Reinforcement Learning with Python (Book)](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)
- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)

## Contact

For any questions or discussions, feel free to reach out via GitHub Issues or email.

