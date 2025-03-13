#!/bin/sh

echo "Starting all Ablation Studies in Parallel..."

python3 ablation-discount/ablation_discount.py &
python3 ablation-learning-rate/ablation_learning_rate.py &
python3 ablation-exploration/ablation_exploration.py &
python3 ablation-nn-size/ablation_nn_size.py &

# Wait for all background processes to finish
wait

echo "All ablation studies completed!"
