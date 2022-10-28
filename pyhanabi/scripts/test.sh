#!/bin/bash

python selfplay.py \
    --save_dir exps/test \
    --num_thread 1 \
    --num_game_per_thread 1 \
    --seed 0 \
    --train_device cuda:0 \
    --act_device cuda:1 \
    --pool pools/sadpool1.json \
