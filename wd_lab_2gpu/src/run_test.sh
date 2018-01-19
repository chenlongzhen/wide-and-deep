#!/bin/bash

rm -rf ../model_dir/*
#CUDA_VISIBLE_DEVICES=0  python wide_n_deep_layer_train.py &

##使用GPU0
export CUDA_VISIBLE_DEVICES=1
nohup  python train.py  > train1.log 2>&1  &
#使用GPU1
#tail -f widetest.log
