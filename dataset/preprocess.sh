#!/usr/bin/env bash
# -*- coding: utf-8 -*-

DATASET=$1
DATA_DIR=/Users/mehdi/Desktop/workspace
DEV_SHUFFLE=0.5

REPO=/Users/mehdi/Desktop/workspace/Research/p-22-ner-position-bias
export PYTHONPATH="$PYTHONPATH:$REPO"


python ${REPO}/dataset/preprocess.py \
--data_dir=${DATA_DIR} \
--dataset=${DATASET} \
--dev_shuffle=${DEV_SHUFFLE}