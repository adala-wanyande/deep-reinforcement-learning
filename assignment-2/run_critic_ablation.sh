#!/bin/bash

EPISODES=1000
ACTOR_LR=0.001
CRITIC_LR=0.001

for dims in "[64]" "[128]" "[128,128]" "[128,64]" "[256,128]" "[256,256]" "[256,128,64]"; do
    filename="data/actor_critic_dims_${dims//[[:blank:]]/}.csv"
    echo "Running critic_dims=$dims"
    python actor_critic_capacity.py \
        --critic-dims "$dims" \
        --actor-lr $ACTOR_LR \
        --critic-lr $CRITIC_LR \
        --episodes $EPISODES \
        --save-path "$filename"
done
