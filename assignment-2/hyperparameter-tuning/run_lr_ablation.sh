#!/bin/bash

ACTOR_LRS=("1e-3" "3e-4" "1e-4")
CRITIC_LRS=("1e-3" "3e-4" "1e-4")

for actor_lr in "${ACTOR_LRS[@]}"; do
  for critic_lr in "${CRITIC_LRS[@]}"; do
    echo "Running actor_lr=$actor_lr, critic_lr=$critic_lr"
    python actor_critic_tune.py --actor-lr $actor_lr --critic-lr $critic_lr
  done
done