#!/bin/bash

python3 train_cifar_act_error.py --model resnet50 -d 0 --dataset cifar10 \
    --out rn50_autoquant \
    --grad_approx mean 100 10 \
    --ae_autoquant 0 CNV B1ir,R1234rs --ae_autoquant 0 BN BR1234c
