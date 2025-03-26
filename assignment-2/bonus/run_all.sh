#!/bin/bash

echo "Starting all policy gradient experiments..."

# Make sure the script stops if any command fails
set -e

# Run training scripts
echo "Running REINFORCE..."
python3 reinforce.py

echo "Running Actor-Critic (AC)..."
python3 actor_critic.py

echo "Running Advantage Actor-Critic (A2C)..."
python3 a2c.py

echo "Running Asynchronous Actor-Critic (A3C)..."
python3 a3c.py

# Run the plotting script
echo "Generating comparison plot..."
python3 plot_all_algorithms.py

echo "All experiments completed and comparison plot saved to plots/comparison_plot.png"