#!/bin/bash

for lr in $( seq 0.02 0.02 0.1); do
    python wetie_3rd.py --learning_rate $lr 
    done

