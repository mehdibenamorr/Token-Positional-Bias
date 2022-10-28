#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#Training ARGS
SEED=23456
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=64
MAX_LENGTH=512
#SEQ_LENTGH=max
OPTIMIZER=adamw_hf
LR_SCHEDULE=linear
LR=5e-5
MAX_EPOCH=5
EVAL_STRATEGY=steps
PADDING=longest
PADDING_SIDE=right
POS_EMB_TYPE=$2

#Script ARGS
EXPERIMENT=bert_position_bias_synthetic
MODEL=bert-base-uncased
DATASET=$1
NBRUNS=1



OUTPUT_DIR=/data/.position_bias

REPO=/data/p-22-ner-position-bias
export PYTHONPATH="$PYTHONPATH:$REPO"


python ${REPO}/experiments/bert_position_bias.py \
--output_dir=${OUTPUT_DIR} \
--dataset=${DATASET} \
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
--position_embedding_type ${POS_EMB_TYPE} \
--include_inputs_for_metrics \
--duplicate \
--truncation

