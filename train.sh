#!/bin/bash

diagnosis=$1
mil=$2
pretrain=$3
mode=$4
gpuN=$5

echo "==================================================="
echo "diagnosis : $diagnosis"
echo "milType   : $mil"
echo "pretrain  : $pretrain"
echo "mode      : $mode"
echo "==================================================="

python3 ./train.py \
    --mil $mil \
    --mode $mode \
    --pretrain $pretrain\
    --diagnosis $diagnosis \
    --gpuN $gpuN