#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# List of Datasets
DATASETS=("conll03" "ontonotes5" "en_ewt" "tweebank")
# List of Models
MODELS=("bert-base-uncased" "google/electra-base-discriminator" "nghuyong/ernie-2.0-base-en" "zhiheng-huang/bert-base-uncased-embedding-relative-key" "zhiheng-huang/bert-base-uncased-embedding-relative-key-query" "bigscience/bloom-560m" "gpt2" "gpt2-large")

# Evaluation
for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo "Evaluating ${MODEL} on ${DATASET}"
    bash experiments/scripts/position_bias_duplicate_no_cv.sh ${DATASET} ${MODEL}
  done
done