#!/usr/bin/env bash
# -*- coding: utf-8 -*-

DATASET=$1
DATA_DIR=/home/mehdi/Desktop/Workspace/CAROLL/ner_datasets
FORMAT=$2
REPO=/home/mehdi/Desktop/Workspace/CAROLL/p-22-ner-position-bias
export PYTHONPATH="$PYTHONPATH:$REPO"


python ${REPO}/dataset/preprocess.py \
--data_dir=${DATA_DIR} \
--dataset=${DATASET} \
--format=${FORMAT} \
--duplicate