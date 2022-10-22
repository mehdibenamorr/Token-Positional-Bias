#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#Training ARGS
SEED=23456
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=12
#MAX_LENGTH=128
SEQ_LENTGH=max
OPTIMIZER=adamw_hf
LR_SCHEDULE=linear
LR=5e-5
MAX_EPOCH=5

#Script ARGS
MODEL=bert-base-uncased
DATASET=$1
NBRUNS=1
PADDING=max_length
PADDING_SIDE=right
SHUFFLE=$2
SHUFFLE_DEV=$3
CONCAT=$4
EXPERIMENT=$5

OUTPUT_DIR=/data/.position_bias


REPO=/data/p-22-ner-position-bias
export PYTHONPATH="$PYTHONPATH:$REPO"


python ${REPO}/experiments/bert_position_bias.py \
--output_dir=${OUTPUT_DIR} \
--dataset=${DATASET} \
--experiment=${EXPERIMENT} \
--seq_length=${SEQ_LENTGH} \
--padding=${PADDING} \
--nbruns=${NBRUNS} \
--seed=${SEED} \
--per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
--per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
--optim ${OPTIMIZER} \
--learning_rate ${LR} \
--lr_scheduler_type ${LR_SCHEDULE} \
--num_train_epochs ${MAX_EPOCH} \
--shuffle ${SHUFFLE} \
--shuffle_eval ${SHUFFLE_DEV} \
--concat ${CONCAT} \
--truncation \
