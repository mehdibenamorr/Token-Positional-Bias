#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#Eval ARGS
SEED=23456
MAX_LENGTH=512
BATCH_SIZE=1
PADDING=max_length
PADDING_SIDE=right

#Script ARGS
EXPERIMENT=bert_position_bias_eval
DATASET=$1
MODE=sep

OUTPUT_DIR=/data/.position_bias

REPO=/data/p-22-ner-position-bias
export PYTHONPATH="$PYTHONPATH:$REPO"

CUDA_VISIBLE_DEVICES=0 python ${REPO}/experiments/evaluate_attns.py \
  --wandb_dir=${OUTPUT_DIR} \
  --dataset=${DATASET} \
  --experiment=${EXPERIMENT} \
  --max_length=${MAX_LENGTH} \
  --padding=${PADDING} \
  --padding_side=${PADDING_SIDE} \
  --debugging \
  --batch_size=${BATCH_SIZE} \
  --duplicate_mode ${MODE} \
  --watch_attentions
