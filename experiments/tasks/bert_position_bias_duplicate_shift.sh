#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#Eval ARGS
SEED=23456
MAX_LENGTH=512

PADDING=max_length
PADDING_SIDE=right

#Script ARGS
EXPERIMENT=bert_position_bias_pos_ids
DATASET=$1
MODE=shift


OUTPUT_DIR=/data/.position_bias

REPO=/data/p-22-ner-position-bias
export PYTHONPATH="$PYTHONPATH:$REPO"


python ${REPO}/experiments/evaluate_attns.py \
--wandb_dir=${OUTPUT_DIR} \
--dataset=${DATASET} \
--experiment=${EXPERIMENT} \
--max_length=${MAX_LENGTH} \
--padding=${PADDING} \
--padding_side=${PADDING_SIDE} \
--truncation \
--duplicate \
--duplicate_mode ${MODE}

