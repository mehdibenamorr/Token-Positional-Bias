#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: random_seed.py
# refer to :
# issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/1868
# Please Notice:
# set for trainer: https://pytorch-lightning.readthedocs.io/en/latest/trainer.html
#   from pytorch_lightning import Trainer, seed_everything
#   seed_everything(42)
#   sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
#   model = Model()
#   trainer = Trainer(deterministic=True)

import random
import torch
import numpy as np
from pytorch_lightning import seed_everything
import argparse
from transformers import TrainingArguments, HfArgumentParser


def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser() -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    parser = HfArgumentParser(TrainingArguments, description="argument parser")
    parser.add_argument('-model', type=str, default='bert-base-uncased',
                        help='Name of a specific model previously saved inside "models" folder'
                             ' or name of an HuggingFace model')
    parser.add_argument("--dataset", type=str, help="dataset to use", choices=["conll03", "ontonotes5"])
    parser.add_argument("--seq_length", type=int, default=512,
                        help="The maximum total input sequence length after tokenization. Sequence longer than this "
                             "will be truncated, sequences shorter will be padded.")
    parser.add_argument('--nbruns', default=10, type=int, help='Number of epochs during training')
    parser.add_argument('--preprocess', default='None', help='If set, training will be done on shuffled data',
                        choices=["shuffle", "rand_pad", "left_pad", "right_pad"])
    parser.add_argument('--position_embedding_type', default='absolute',
                        help=' Type of position embedding. Choose one of "absolute", "relative_key", '
                             '"relative_key_query". For positional embeddings use "absolute". For more information '
                             'on "relative_key", please refer to Self-Attention with Relative Position Representations'
                             '(Shaw et al.). For more information on "relative_key_query", please refer to Method 4 in '
                             'Improve Transformer Models with Better Relative Position Embeddings (Huang et al.).',
                        choices=["absolute", "relative_key", "relative_key_query"])
    parser.add_argument("--debugging", action="store_true", help="whether it's debugging")

    return parser


if __name__ == '__main__':
    # without this line, x would be different in every execution.
    set_random_seed(0)

    x = np.random.random()
    print(x)

    parser = get_parser()
    t_args, args = parser.parse_args_into_dataclasses()
    print(t_args, args)
