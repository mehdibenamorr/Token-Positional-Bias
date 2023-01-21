#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: processor.py
#

from transformers import AutoTokenizer
import numpy as np


class Processor(object):
    NAME = "Processor"

    def __init__(self, pretrained_checkpoint, type="ner", max_length=None, lower_case=True, kwargs=None):
        padding_side = kwargs.get("padding_side", "right")
        padding = kwargs.get("padding", "max_length")
        truncation = kwargs.get("truncation", False)
        self.return_truncated_tokens = kwargs.get("return_truncated_tokens", False)

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint, do_lower_case=lower_case,
                                                        padding_side=padding_side, add_prefix_space=True)
        self.resize_token_embeddings = False
        if self._tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a pad token
            self.resize_token_embeddings = True
        self.padding, self.truncation_strategy, self.max_length, _ = self.tokenizer._get_padding_truncation_strategies(
            padding=padding, truncation=truncation,
            max_length=max_length)

        if self.truncation_strategy and self.max_length:
            self.return_truncated_tokens = True

        if type == "ner":
            self.labels_key = "ner_tags"
        elif type == "pos":
            self.labels_key = "pos_tags"

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

        examples_ = self.process_batch(examples, duplicate=duplicate, k=k, mode=duplicate_mode)

        tokenized_inputs = self._tokenizer(examples_["tokens"],
                                           truncation=self.truncation_strategy,
                                           is_split_into_words=True,
                                           max_length=self.max_length,
                                           return_overflowing_tokens=self.return_truncated_tokens,
                                           padding=self.padding)
        labels = []

        j = 0
        for i, label in enumerate(examples_[self.labels_key]):

            word_ids = tokenized_inputs.word_ids(batch_index=j)
            overflowing_tokens = tokenized_inputs.encodings[j].overflowing if hasattr(tokenized_inputs.encodings[j],
                                                                                      "overflowing") else None
            if overflowing_tokens and duplicate:
                items = [tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"],
                             tokenized_inputs["overflow_to_sample_mapping"], tokenized_inputs.encodings]
                if "token_type_ids" in tokenized_inputs:
                    items.append(tokenized_inputs["token_type_ids"])
                lth = len(overflowing_tokens)
                for item in items:
                    del item[j:j + lth + 1]
                continue
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

    def duplicate_seq(self, features, k=2, mode="none", sep_token="[SEP]"):
        if mode == "sep":
            tokens = [(k * (x + [sep_token]))[:-1] for x in features["tokens"]]
            tags = [(k * (x + [-100]))[:-1] for x in features[self.labels_key]]
        else:
            tokens = [k * x for x in features["tokens"]]
            tags = [k * x for x in features[self.labels_key]]
        k_ = [[k] for x in features[self.labels_key]]
        return {"id": [str(i) for i in range(len(tokens))],
                "tokens": tokens,
                self.labels_key: tags,
                "original_tokens": features["tokens"],
                "original_tags": features[self.labels_key],
                "k": k_}

    def process_batch(self, examples, duplicate=False, k=2, mode="none"):
        examples_ =examples
        features_ = examples_.data
        if duplicate:
            if mode in ["none", "sep"]:
                data = self.duplicate_seq(features_, k=k, mode=mode, sep_token="[SEP]")
                examples_.data = data

        return examples_


if __name__ == '__main__':
    checkpoint = "bigscience/bloom-560m"
    # from dataset.ner_dataset import NERDataset
    #
    # ner_dataset = NERDataset(dataset="conll03", debugging=False)
    #
    # ner_processor = Processor(pretrained_checkpoint=checkpoint, type="ner", max_length=512,
    #                           kwargs={"truncation": True, })
    #
    # test_ner = ner_dataset.dataset["test_"].map(ner_processor.tokenize_and_align_labels,
    #                                             fn_kwargs={"duplicate": True, "k": 10,"duplicate_mode": "sep"}, load_from_cache_file=False,
    #                                             batched=True)
    #
    # print("Duplicated Test set: ", test_ner[0])
    #
    # tokenized_datasets = ner_dataset.dataset.map(ner_processor.tokenize_and_align_labels, batched=True)
    #
    # print(ner_dataset)
    #
    # print("*" * 100)
    #
    # print(tokenized_datasets)
    #
    # print("First sample: ", ner_dataset.dataset['train'][0])
    #
    # print("*" * 100)
    #
    # print("First tokenized sample: ", tokenized_datasets['train'][0])

    from dataset.pos_dataset import POSDataset

    pos_dataset = POSDataset(dataset="tweebank", debugging=False)

    pos_processor = Processor(pretrained_checkpoint=checkpoint, type="pos", max_length=512,
                              kwargs={"truncation": True, })

    test_pos = pos_dataset.dataset["test_"].map(pos_processor.tokenize_and_align_labels,
                                                fn_kwargs={"duplicate": True, "k": 10, "duplicate_mode": "sep"},
                                                load_from_cache_file=False,
                                                batched=True)
    print("Duplicated Test set: ", test_pos[0])
    tokenized_datasets = pos_dataset.dataset.map(pos_processor.tokenize_and_align_labels, batched=True)

    print(pos_dataset)

    print("*" * 100)

    print(tokenized_datasets)

    print("First sample: ", pos_dataset.dataset['train'][0])

    print("*" * 100)

    print("First tokenized sample: ", tokenized_datasets['train'][0])
