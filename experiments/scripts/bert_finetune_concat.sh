#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#Training ARGS
SEED=23456
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=64
MAX_LENGTH=512
OPTIMIZER=adamw_hf
LR_SCHEDULE=linear
LR=5e-5
MAX_EPOCH=5
EVAL_STRATEGY=steps
PADDING=max_length
PADDING_SIDE=right
POS_EMB_TYPE=absolute

#Script ARGS
EXPERIMENT=finetune_concat
MODEL=$2
DATASET=$1
NBRUNS=5

OUTPUT_DIR=/data/.finetuning

REPO=/data/p-22-ner-position-bias
export PYTHONPATH="$PYTHONPATH:$REPO"

python ${REPO}/experiments/position_bias.py \
  --output_dir=${OUTPUT_DIR} \
  --model="${MODEL}" \
  --dataset="${DATASET}" \
  --experiment=${EXPERIMENT} \
  --max_length=${MAX_LENGTH} \
  --padding=${PADDING} \
  --padding_side=${PADDING_SIDE} \
  --nbruns=${NBRUNS} \
  --seed=${SEED} \
  --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --optim ${OPTIMIZER} \
  --learning_rate ${LR} \
  --lr_scheduler_type ${LR_SCHEDULE} \
  --num_train_epochs ${MAX_EPOCH} \
  --evaluation_strategy ${EVAL_STRATEGY} \
  --save_strategy ${EVAL_STRATEGY} \
  --logging_strategy ${EVAL_STRATEGY} \
  --position_embedding_type ${POS_EMB_TYPE} \
  --concatenate
