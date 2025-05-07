#!/bin/bash

# Ensure output directory exists
mkdir -p data

# List of hidden layer sizes to test
hidden_sizes=(64 128 256 512)

for size in "${hidden_sizes[@]}"
do
  echo "Running PPO with hidden size $size..."
  python3 ppo_ablation.py --hidden $size
done

echo "All ablation experiments completed."
