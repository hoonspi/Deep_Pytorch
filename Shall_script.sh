#!/bin/bash

for lr in $( seq 0.02 0.02 0.1); do
    python wetie_3rd.py --learning_rate $lr 
    done
for epoch in $(seq 10 20 30); do
    python wetie_3rd.py --num_epoch $epoch 
    done
for hidden in $(seq 64 128 256); do
    python wetie_3rd.py --hidden1 $hidden --hidden2 $hidden --hidden3 $hidden 
    done     
