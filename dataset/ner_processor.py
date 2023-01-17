#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: ner_processor.py
#

import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification


class NERProcessor(object):
    NAME = "NERProcessor"

    def __init__(self, pretrained_checkpoint, max_length=None, lower_case=True, kwargs=None):
        padding_side = kwargs.get("padding_side", "right")
        padding = kwargs.get("padding", "longest")
        truncation = kwargs.get("truncation", False)

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint, do_lower_case=lower_case,
                                                        padding_side=padding_side)
        self.padding, self.truncation_strategy, self.max_length, _ = self.tokenizer._get_padding_truncation_strategies(
            padding=padding, truncation=truncation,
            max_length=max_length)
        self.return_truncated_tokens = kwargs.get("return_truncated_tokens", False)
        if self.truncation_strategy and self.max_length:
            self.return_truncated_tokens = True

    @property
    def tokenizer(self):
        return self._tokenizer

    def tokenize_and_align_labels(self,
                                  examples,
                                  label_all_tokens=True, duplicate=False, k=2,
                                  duplicate_mode="none"):

        def align_label(labels, word_ids, label_all_tokens=True):
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(labels[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            return label_ids

        examples_ = process_batch(examples, duplicate=duplicate, k=k, mode=duplicate_mode)

        tokenized_inputs = self._tokenizer(examples_["tokens"],
                                           truncation=self.truncation_strategy,
                                           is_split_into_words=True,
                                           max_length=self.max_length,
                                           return_overflowing_tokens=self.return_truncated_tokens,
                                           padding=self.padding)
        labels = []

        j = 0
        for i, label in enumerate(examples_[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=j)
            overflowing_tokens = tokenized_inputs.encodings[j].overflowing if hasattr(tokenized_inputs.encodings[j],
                                                                                      "overflowing") else None
            label_ids = align_label(label, word_ids, label_all_tokens=label_all_tokens)
            labels.append(label_ids)
            if overflowing_tokens:
                for encoding in overflowing_tokens:
                    word_ids = encoding.word_ids
                    label_ids = align_label(label, word_ids, label_all_tokens)
                    labels.append(label_ids)
                    j += 1
            j += 1

        # Extract mapping between new and old indices
        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping", None)
        if sample_map is not None:
            for key, values in examples_.items():
                tokenized_inputs[key] = [values[i] for i in sample_map]
        tokenized_inputs["labels"] = labels

        if duplicate and duplicate_mode == "shift":
            pos_ids = [i for i in range(self.max_length)]
            position_ids = []
            for i, input_ids in enumerate(tokenized_inputs["input_ids"]):
                (indices,) = np.array(input_ids).nonzero()
                indices = indices[1:]
                shifted_pos = np.array(pos_ids)
                shifted_pos[indices] += (k - 1) * len(indices)
                position_ids.append(shifted_pos.tolist())
            tokenized_inputs["position_ids"] = position_ids
        return tokenized_inputs


def duplicate_seq(features, k=2, mode="none", sep_token="[SEP]"):
    if mode == "sep":
        tokens = [(k * (x + [sep_token]))[:-1] for x in features["tokens"]]
        tags = [(k * (x + [-100]))[:-1] for x in features["ner_tags"]]
    else:
        tokens = [k * x for x in features["tokens"]]
        tags = [k * x for x in features["ner_tags"]]
    k_ = [[k] for x in features["ner_tags"]]
    return {"id": [str(i) for i in range(len(tokens))],
            "tokens": tokens,
            "ner_tags": tags,
            "original_tokens": features["tokens"],
            "original_tags": features["ner_tags"],
            "k": k_}


def process_batch(examples, duplicate=False, k=2, mode="none"):
    features_ = examples.data
    if duplicate:
        if mode in ["none", "sep"]:
            data = duplicate_seq(features_, k=k, mode=mode, sep_token="[SEP]")
            examples.data = data

    return examples


if __name__ == '__main__':
    from dataset.ner_dataset import NERDataset

    checkpoint = "bert-base-uncased"
    conll03 = NERDataset(dataset="conll03", debugging=True)

    ner_processor = NERProcessor(pretrained_checkpoint=checkpoint, max_length=512, kwargs={})

    for k in range(1, 11):
        test_dataset = conll03.dataset["test_"].map(ner_processor.tokenize_and_align_labels,
                                                    fn_kwargs={"duplicate": True, "k": k}, load_from_cache_file=False,
                                                    batched=True)
        print(test_dataset[0])

    tokenized_datasets = conll03.dataset.map(ner_processor.tokenize_and_align_labels, batched=True)

    print(conll03)

    print("*" * 100)

    print(tokenized_datasets)

    print("First sample: ", conll03.dataset['train'][0])

    print("*" * 100)

    print("First tokenized sample: ", tokenized_datasets['train'][0])
