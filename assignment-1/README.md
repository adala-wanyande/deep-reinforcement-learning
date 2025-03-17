# Assignment 1: Deep Q-Learning and Ablation Studies

## Overview

This repository contains an extensive analysis of **Deep Q-Networks (DQN)** in the **CartPole-v1** environment. The experiments include **hyperparameter ablations** and **stabilization studies** for improving training performance. The results demonstrate the impact of various techniques such as **experience replay** and **target networks** on DQN stability and performance.

## Repository Structure

```
assignment-1/
│── naive-dqn/                 # Naive DQN experiments (no stabilizations)
│   ├── ablation-discount/     # Discount factor tests
│   ├── ablation-exploration/  # Exploration factor tests
│   ├── ablation-learning-rate/ # Learning rate tests
│   ├── ablation-nn-size/      # Neural network size tests
│   ├── best-hyperparameters/  # Best identified hyperparameters for Naive DQN
│   ├── data/                  # CSV results for naive DQN ablation tests
│── stable-dqn/                # Stabilized DQN experiments
│   ├── ablation-experience-replay/  # Experience replay tests
│   ├── ablation-target-network/     # Target network tests
│   ├── ablation-both/               # Both experience replay & target networks
│   ├── data/                        # CSV results for stable DQN
│── visuals/                # Directory containing generated plots
│── plot_all.py             # Script for visualizing and comparing experiments
│── run_ablation_studies.sh # Shell script for running all experiments
│── README.md               # This guide
│── assignment_description.pdf  # Original assignment instructions
```

## How to Run Experiments

### Setting Up the Environment
Ensure you have Python installed and set up the required dependencies using:
```bash
pip install -r requirements.txt
```

### Running All DQN Ablation Studies At Once
```bash
bash run_ablation_studies.sh
```

### Running Individual Hyperparameter Ablation Studies (Naive DQN)
To run individual ablation studies for **learning rate, discount factor, exploration, and network size**, navigate to the respective folders and execute:
```bash
python naive-dqn/ablation-learning-rate/ablation_learning_rate.py
```
Repeat this for **discount, exploration, and nn-size** experiments.

### Running Individual Stabilized DQN Experiments
To run experiments testing **experience replay**, **target networks**, and their combination, execute:
```bash
python stable-dqn/ablation-experience-replay/ablation_experience_replay.py
python stable-dqn/ablation-target-network/ablation_target_network.py
python stable-dqn/ablation-both/ablation_both.py
```

### Visualizing Results
All results are stored as CSV files in `naive-dqn/data/` and `stable-dqn/data/`. To generate plots for all experiments, run:
```bash
python plot_all.py
```
The plots will be saved in the `visuals/` directory.

## Expected Outcomes
- **Naive DQN**: Poor stability, slow learning, and high variance.
- **Experience Replay**: Smoother learning curves, improved efficiency.
- **Target Networks**: Reduced variance, but slower learning.
- **Experience Replay + Target Networks**: Best overall performance with the most stable convergence.

## Notes for Reviewers
- The `data/` directory contains all experimental results for verification.
- The `visuals/` directory includes plots for each ablation study.
- The `best-hyperparameters/` folder stores the optimal hyperparameters identified.
- The `plot_all.py` script generates comparison plots for quick analysis.

## Contact
For any clarifications, please reach out at **b.a.wanyande@umail.leidenuniv.nl**.

