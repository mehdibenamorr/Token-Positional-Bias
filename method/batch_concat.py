import random
from itertools import permutations
from operator import itemgetter
import torch.nn.functional as F
import torch
import numpy as np


def unique_permutations(sequence, r, n):
    """Yield n unique permutations of r elements from sequence"""
    seen = set()
    while len(seen) < n:
        # This line of code adapted from Pablo Ruiz's answer:
        candidate_permutation = tuple(np.random.choice(sequence, r, replace=False))

        if candidate_permutation not in seen:
            seen.add(candidate_permutation)
            yield candidate_permutation


def concatenate_batch(batch, max_length=512):
    input_ids = list(batch["input_ids"])
    labels = list(batch["labels"])
    dtype = batch["input_ids"].dtype
    # Get the original sequence
    original_input_idx = list(map(lambda x: x.nonzero()[1:].squeeze(), input_ids))
    original_input_ids = list(map(lambda x: x[x.nonzero()[1:]].squeeze(), input_ids))
    original_labels = [labels[i][original_input_idx[i]] for i in range(len(original_input_idx))]
    seq_lengths = list(map(lambda x: x.shape[0], original_input_ids))

    # Get subsets of sequence that add up to max_length
    total_length = 0
    seq_idx = []
    to_concat_idx = []
    for i, length in enumerate(seq_lengths):
        if total_length + length <= max_length - 1:
            seq_idx.append(i)
            total_length += length
        else:
            to_concat_idx.append(seq_idx)
            seq_idx = [i]
            total_length = length
    if seq_idx:
        to_concat_idx.append(seq_idx)

    # Concatenate the sequences and permutate in random orders
    new_input_ids = []
    new_labels = []
    for subset in to_concat_idx:
        # Concatenate the sequences
        if len(subset) == 1:
            new_length = sum([seq_lengths[i] for i in subset]) + 1
            new_input_ids += [
                F.pad(torch.cat([torch.Tensor([101]).to(dtype=dtype), original_input_ids[subset[0]]], dim=0),
                      pad=(0, max_length - new_length), mode="constant", value=0)]
            new_labels += [
                F.pad(torch.cat([torch.Tensor([-100]).to(dtype=dtype), original_labels[subset[0]]], dim=0),
                                 pad=(0, max_length - new_length), mode="constant", value=-100)]

        else:
            new_length = sum([seq_lengths[i] for i in subset]) + 1
            permutations_list = list(unique_permutations(subset, len(subset), len(subset)))
            new_input_ids += [
                F.pad(torch.cat(
                    [torch.Tensor([101]).to(dtype=dtype)] + list(itemgetter(*permutation)(original_input_ids)),
                    dim=0), pad=(0, max_length - new_length), mode="constant", value=0)
                for permutation in permutations_list]
            new_labels += [
                F.pad(
                    torch.cat([torch.Tensor([-100]).to(dtype=dtype)] + list(itemgetter(*permutation)(original_labels)),
                              dim=0), pad=(0, max_length - new_length), mode="constant", value=-100)
                for permutation in permutations_list]

    batch["input_ids"] = torch.stack(new_input_ids)
    batch["labels"] = torch.stack(new_labels)
    batch["attention_mask"] = torch.where(batch["input_ids"] != 0, torch.ones_like(batch["input_ids"]),
                                          torch.zeros_like(batch["input_ids"]))
    return batch
