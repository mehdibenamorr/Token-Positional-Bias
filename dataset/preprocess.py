#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: preprocess.py
#
from utils import set_random_seed

set_random_seed(23456)
import argparse
import os
import random
from tqdm import tqdm


def tags_to_spans(tokens, tags):
    """
    Desc:
        get from token_level labels to list of entities,
        it doesnot matter tagging scheme is BMES or BIO or IOBES
    Returns:
        a list of entities
        [[token1],[token2,token3],...],
        [[tag1],[tag2,tag3],...]
    """

    span_labels = []
    segmented_tokens = []
    segmented_labels = []
    last = "O"
    start = -1
    for i, tag in enumerate(tags):
        pos, _ = (None, "O") if tag == "O" else tag.split("-")
        if (pos == "S" or pos == "B" or tag == "O") and last != "O":
            span_labels.append((start, i - 1, last.split("-")[-1]))
            segmented_tokens.append(tokens[start:i])
            segmented_labels.append(tags[start:i])
        if pos == "B" or pos == "S" or last == "O":
            start = i
        if tag == "O" and last == "O":
            segmented_tokens.append([tokens[i]])
            segmented_labels.append([tags[i]])
        last = tag

    if tags[-1] != "O":
        span_labels.append((start, len(tags) - 1, tags[-1].split("-"[-1])))
        segmented_tokens.append(tokens[start:len(tags)])
        segmented_labels.append(tags[start:len(tags)])

    return segmented_tokens, segmented_labels


def shuffle_dataset(dataset, ratio=0.2):
    shuffled = dict()
    ids = dataset.keys()
    n = int(ratio * len(ids))
    to_shuffle = random.sample(ids, n) if ratio < 1.0 else ids
    for id, item in tqdm(dataset.items()):
        if id in to_shuffle:
            segmented_tokens, segmented_labels = tags_to_spans(item["tokens"], item["ner_tags"])
            indices = list(range(len(segmented_tokens)))
            random.shuffle(indices)
            shuffled_item = {"tokens": [], "ner_tags": []}
            for idx in indices:
                shuffled_item["tokens"] += segmented_tokens[idx]
                shuffled_item["ner_tags"] += segmented_labels[idx]
            shuffled.update({id: shuffled_item})
        else:
            shuffled.update({id: item})
    return shuffled


def process_data(filepath, ratio=None):
    # The full dataset split (val, or test)
    dataset = dict()
    with open(filepath, encoding="utf-8") as f:
        guid = 0
        tokens = []
        ner_tags = []
        lines = f.readlines()
        for line in lines:
            if line == "" or line == "\n":
                if tokens:
                    dataset.update({str(guid): {
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
                    })
                    guid += 1
                    tokens = []
                    ner_tags = []
            else:
                splits = line.split(" ")
                tokens.append(splits[0])
                ner_tags.append(splits[1].rstrip())
    # Shuffling of tokens
    shuffled = shuffle_dataset(dataset, ratio=ratio)
    save(shuffled, filename=filepath)
    return shuffled


def save(shuffled, filename):
    with open(filename + ".shuffled", "w") as file:
        for id, item in shuffled.items():
            for i in range(len(item["tokens"])):
                file.write(" ".join([item["tokens"][i], item["ner_tags"][i]]))
                file.write("\n")
            file.write("\n")


def processing_args() -> argparse.ArgumentParser:
    """
        return basic arg parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="dataset directory")
    parser.add_argument("--dataset", type=str, help="dataset to use", choices=["en_conll03", "ontonotes5"])
    parser.add_argument("--dev_shuffle",
                        default=0.5,
                        help="percentage of data samples to be shuffled in the validation set")

    return parser


if __name__ == "__main__":
    parser = processing_args()
    args = parser.parse_args()
    # ### Train dataset
    train_path = os.path.join(args.data_dir, args.dataset, "train.word.iobes")
    process_data(train_path, ratio=0.5)

    # ### Dev dataset
    print(args.data_dir)
    dev_path = os.path.join(args.data_dir, args.dataset, "dev.word.iobes")
    process_data(dev_path, ratio=0.5)
    ### Test dataset
    test_path = os.path.join(args.data_dir, args.dataset, "test.word.iobes")
    process_data(test_path, ratio=1)
