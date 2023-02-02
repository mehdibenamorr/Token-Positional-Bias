#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# List of Datasets
DATASETS=("conll03" "ontonotes5" "en_ewt" "tweebank")
# List of Models
MODEL=bert-base-uncased
# List of methods
METHODS=("shift" "concat")

# Evaluation
for DATASET in "${DATASETS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    echo "Finetuning ${MODEL} on ${DATASET} with method ${METHOD}"
    # shellcheck disable=SC1072
    if [ ${METHOD} == "shift" ]; then
      bash experiments/scripts/bert_finetune_shift.sh ${DATASET} ${MODEL}
    elif [ ${METHOD} == "concat" ]; then
      bash experiments/scripts/bert_finetune_concat.sh ${DATASET} ${MODEL}
    fi
  done
done
