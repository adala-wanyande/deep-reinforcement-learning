#!/bin/bash

# Exit if any command fails
set -e

echo "Running REINFORCE..."
python reinforce.py

echo "Running Actor-Critic..."
python ac.py

echo "Running A2C..."
python a2c.py

echo "All training scripts completed."

echo "Generating plot..."
python plot_all.py