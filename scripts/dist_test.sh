#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=${PORT:-47821}

# usage
if [ $# -ne 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_test.sh [number of gpu] [path to option file]"
    exit
fi

# PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    basicsr/test.py -opt $CONFIG --launcher pytorch
