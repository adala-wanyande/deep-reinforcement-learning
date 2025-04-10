#!/bin/bash

# Ensure script exits on first failure
set -e

# Create data and plots folders if they don't exist
mkdir -p data plots

# Run CartPole experiments (REINFORCE, AC, A2C)
echo "Running CartPole experiments..."
python3 reinforce.py
python3 ac.py
python3 a2c.py

# Run Acrobot experiments (REINFORCE, AC, A2C)
echo "Running Acrobot experiments..."
python3 reinforce_acrobot.py
python3 ac_acrobot.py
python3 a2c_acrobot.py

# Generate plots for both environments
echo "Generating plots..."
python3 plot_all.py
python3 plot_all_bonus.py

echo "All experiments completed and plots generated."