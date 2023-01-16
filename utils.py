#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: utils.py
# refer to :
# issue: https://pytorch.org/docs/stable/notes/randomness.html#reproducibility


import random
import torch
import numpy as np
import argparse
from transformers import TrainingArguments, HfArgumentParser, set_seed


def set_random_seed(seed: int):
    """set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser(HF=True) -> argparse.ArgumentParser:
    """
    return basic arg parser
    """
    if HF:
        parser = HfArgumentParser(TrainingArguments, description="argument parser")
    else:
        parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help='Name of a specific model previously saved inside "models" folder'
                             ' or name of an HuggingFace model')
    parser.add_argument('--wandb_user', type=str, default='benamor',
                        help='Name of the WandB account where the models are stored')
    parser.add_argument('--wandb_dir', type=str, default='.wandb',
                        help='Local directory of the WandB where the logs are stored')
    parser.add_argument('--experiment', type=str, default='bert-position-bias',
                        help='Name of the experiment')
    parser.add_argument("--dataset", type=str, help="dataset to use",
                        choices=["conll03", "ontonotes5", "en_ewt", "tweebank"])
    parser.add_argument("--max_length", type=int, default=512,
                        help="The maximum total input sequence length to overwrite seq_length by given number. "
                             "Sequence longer than this "
                             "will be truncated, sequences shorter will be padded.")
    parser.add_argument("--truncation", action="store_true",
                        help="Whether to truncate sequences exceeding max_length")
    parser.add_argument("--return_truncated_tokens", action="store_true",
                        help="Whether to return truncated tokens within the encoded batch")
    parser.add_argument("--padding_side", type=str, default="right", choices=["right", "left", "random"],
                        help="The padding strategy to be used. 'random' means pad tokens will be injected within "
                             "original sequence in random positions")
    parser.add_argument("--padding", type=str, default=None, choices=["max_length", "longest", "do_not_pad", None],
                        help="The padding strategy to be used. 'random' means pad tokens will be injected within "
                             "original sequence in random positions")
    parser.add_argument('--nbruns', default=10, type=int, help='Number of epochs during training')
    parser.add_argument('--cv', default=10, type=int, help='Number fold in k-fold cross validation')
    parser.add_argument('--concatenate', action="store_true",
                        help='If set, sequences are concatenated in batches to max_length',
                        )
    parser.add_argument('--position_shift', action="store_true",
                        help='If set, position ids are shifted to a random position',
                        )
    parser.add_argument('--duplicate', action="store_true",
                        help='If set, test set will be duplicated',
                        )
    parser.add_argument('--duplicate_mode', type=str, default="sep", choices=["none", "shift", "sep"],
                        help='Mode of duplication: possible values are, "none", "shift" (position_ids), '
                             'and "sep" (adding [SEP] token))',
                        )
    parser.add_argument('--watch_attentions', action="store_true",
                        help='If set, attention scores will be logged',
                        )
    parser.add_argument('--batch_size', default=64, type=int, help='batch size when evaluating')
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
