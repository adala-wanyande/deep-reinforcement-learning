#!/bin/sh

echo "Starting all Ablation Studies in Sequence..."

# Run each ablation study and wait for completion before moving to the next
echo "Running Discount Factor Ablation..."
python3 ablation-discount/ablation_discount.py
echo "Generating Discount Plot..."
python3 ablation-discount/plot_discount.py

echo "Running Learning Rate Ablation..."
python3 ablation-learning-rate/ablation_learning_rate.py
echo "Generating Learning Rate Plot..."
python3 ablation-learning-rate/plot_learning_rate.py

echo "Running Exploration Ablation..."
python3 ablation-exploration/ablation_exploration.py
echo "Generating Exploration Plot..."
python3 ablation-exploration/plot_exploration.py

echo "Running Neural Network Size Ablation..."
python3 ablation-nn-size/ablation_nn_size.py
echo "Generating Neural Network Size Plot..."
python3 ablation-nn-size/plot_nn.py

echo "All ablation studies and plots completed!"
