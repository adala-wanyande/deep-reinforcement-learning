#!/bin/bash

# Set error handling
set -e  # Exit if any command fails

# Define paths
NAIVE_DQN_DIR="./naive-dqn"
STABLE_DQN_DIR="./stable-dqn"

# Create output directories if they don't exist
mkdir -p "$NAIVE_DQN_DIR/data"
mkdir -p "$STABLE_DQN_DIR/data"

echo "===================================="
echo " Running Naive DQN Ablation Studies"
echo "===================================="

# Run naive DQN ablation studies
echo "Running Learning Rate Ablation..."
python3 "$NAIVE_DQN_DIR/ablation-learning-rate/ablation_learning_rate.py"

echo "Running Discount Factor Ablation..."
python3 "$NAIVE_DQN_DIR/ablation-discount/ablation_discount.py"

echo "Running Exploration Rate Ablation..."
python3 "$NAIVE_DQN_DIR/ablation-exploration/ablation_exploration.py"

echo "Running Neural Network Size Ablation..."
python3 "$NAIVE_DQN_DIR/ablation-nn-size/ablation_nn_size.py"

# Run best hyperparameter experiment for Naive DQN
echo "Running Naive DQN with Best Hyperparameters..."
python3 "$NAIVE_DQN_DIR/best-hyperparameters/best_hyperparameters.py"

echo "======================================"
echo " Running Stable DQN Ablation Studies"
echo "======================================"

# Run stable DQN ablation studies
echo "Running Experience Replay Ablation..."
python3 "$STABLE_DQN_DIR/ablation-experience-replay/ablation_experience_replay.py"

echo "Running Target Network Ablation..."
python3 "$STABLE_DQN_DIR/ablation-target-network/ablation_target_network.py"

echo "Running Combined Stabilization (Target Network + Experience Replay)..."
python3 "$STABLE_DQN_DIR/ablation-both/ablation_both.py"

echo "=========================================="
echo " All Experiments Completed Successfully!"
echo "=========================================="

# Notify user
echo "Results saved in respective 'data' directories. Use plotting scripts in 'plot_all.py' for visualizations."
