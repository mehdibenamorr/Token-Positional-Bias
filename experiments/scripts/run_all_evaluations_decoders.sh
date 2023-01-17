#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# List of Datasets
DATASETS=("conll03" "ontonotes5")
# List of Models
MODELS=( "bigscience/bloom-560m" "gpt2" "gpt2-large")

# Evaluation
for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Evaluating ${MODEL} on ${DATASET}"
    bash experiments/scripts/position_bias_duplicate_no_cv.sh ${DATASET} ${MODEL}
  done
done