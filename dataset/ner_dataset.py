#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: ner_dataset.py
#
# Ref: https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py

import os
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
from datasets import ClassLabel, DownloadConfig


class NERDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for NERDataset (CoNll, ontonotes5)"""

    def __init__(self, **kwargs):
        """BuilderConfig for NERDataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NERDatasetConfig, self).__init__(**kwargs)


logger = datasets.logging.get_logger(__name__)

_CITATION = ""
_DESCRIPTION = """\
"""

CONFIGS = {"conll03": {"version": datasets.Version("0.1.0"), "description": "CoNLL dataset"},
           "ontonotes5": {"version": datasets.Version("0.1.0"), "description": "Ontonotes5 dataset"}}

URLS = {"conll03": "https://share.pads.fim.uni-passau.de/datasets/nlp/en_conll03/",
        "ontonotes5": "https://share.pads.fim.uni-passau.de/datasets/nlp/ontonotes5/"}


class NERDatasetbuilder(datasets.GeneratorBasedBuilder):
    """NER dataset (conll03, ontonotes5)."""

    BUILDER_CONFIGS = []

    def __init__(self,
                 *args,
                 cache_dir,
                 dataset="conll03",
                 train_file="train.word.iobes",
                 dev_file="dev.word.iobes",
                 test_file="test.word.iobes",
                 all_file="all.word.iobes",
                 debugging=False,
                 **kwargs):
        self._ner_tags = self.get_labels(dataset)
        suffix_ = "_debug" if debugging else ""
        self.BUILDER_CONFIGS = [NERDatasetConfig(name=dataset + suffix_, version=CONFIGS[dataset]["version"],
                                                 description=CONFIGS[dataset]["description"])]
        self._url = URLS[dataset]
        self._train_file = train_file
        self._train_file_ = f"{train_file}.cut"
        self._dev_file = dev_file
        self._dev_file_ = f"{dev_file}.cut"
        self._test_file = test_file
        self._test_file_shuffled = f"{test_file}.shuffled"
        self._test_file_ = f"{test_file}.cut"
        self._all_file = f"{all_file}.cut"
        self.debugging = debugging
        self.limit = 100
        super(NERDatasetbuilder, self).__init__(*args, cache_dir=cache_dir, **kwargs)

    @classmethod
    def get_labels(cls, dataset):
        """gets the list of labels for this data set."""
        if dataset == "ontonotes5":
            return ['O', 'S-PERSON', 'B-PERSON', 'I-PERSON', 'E-PERSON', 'S-GPE', 'B-GPE', 'I-GPE', 'E-GPE',
                    'S-ORDINAL', 'B-ORDINAL', 'I-ORDINAL', 'E-ORDINAL', 'S-NORP', 'B-NORP', 'I-NORP', 'E-NORP',
                    'S-CARDINAL', 'B-CARDINAL', 'I-CARDINAL', 'E-CARDINAL', 'S-ORG', 'B-ORG', 'I-ORG', 'E-ORG',
                    'S-DATE', 'B-DATE', 'I-DATE', 'E-DATE', 'S-LOC', 'B-LOC', 'I-LOC', 'E-LOC', 'S-EVENT', 'B-EVENT',
                    'I-EVENT', 'E-EVENT', 'S-TIME', 'B-TIME', 'I-TIME', 'E-TIME', 'S-PRODUCT', 'B-PRODUCT', 'I-PRODUCT',
                    'E-PRODUCT', 'S-LANGUAGE', 'B-LANGUAGE', 'E-LANGUAGE', 'I-LANGUAGE', 'S-WORK_OF_ART',
                    'B-WORK_OF_ART', 'I-WORK_OF_ART', 'E-WORK_OF_ART', 'S-FAC', 'B-FAC', 'I-FAC', 'E-FAC', 'S-MONEY',
                    'B-MONEY', 'I-MONEY', 'E-MONEY', 'S-QUANTITY', 'B-QUANTITY', 'I-QUANTITY', 'E-QUANTITY', 'S-LAW',
                    'B-LAW', 'I-LAW', 'E-LAW', 'S-PERCENT', 'B-PERCENT', 'I-PERCENT', 'E-PERCENT']
        elif dataset == "conll03":
            return ["O", "S-MISC", "B-MISC", "I-MISC", "E-MISC", "S-ORG", "B-ORG", "I-ORG", "E-ORG", "S-LOC", "B-LOC",
                    "I-LOC", "E-LOC", "S-PER", "B-PER", "I-PER", "E-PER"]
        raise ValueError(f"Dataset {dataset} is not configured. Available datasets are {CONFIGS.keys()}.")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(self._ner_tags))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{self._url}{self._train_file}",
            "train*": f"{self._url}{self._train_file_}",
            "dev": f"{self._url}{self._dev_file}",
            "dev*": f"{self._url}{self._dev_file_}",
            "test": f"{self._url}{self._test_file}",
            "test*": f"{self._url}{self._test_file_}",
            "all*": f"{self._url}{self._all_file}",
            "test-shuffled": f"{self._url}{self._test_file_shuffled}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
            datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="train_"),
                                    gen_kwargs={"filepath": downloaded_files["train*"]}),
            datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="dev_"),
                                    gen_kwargs={"filepath": downloaded_files["dev*"]}),
            datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="test_"),
                                    gen_kwargs={"filepath": downloaded_files["test*"]}),
            datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="all_"),
                                    gen_kwargs={"filepath": downloaded_files["all*"]}),
            datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="shuffled_test"),
                                    gen_kwargs={"filepath": downloaded_files["test-shuffled"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                        if self.debugging and guid >= self.limit:
                            return
                else:
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }


class NERDataset(object):
    """
    """
    NAME = "NERDataset"

    def __init__(self, dataset="conll03", debugging=False):
        cache_dir = os.path.join(str(Path.home()), '.ner_datasets')
        print("Cache directory: ", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        download_config = DownloadConfig(cache_dir=cache_dir)
        self._dataset = NERDatasetbuilder(cache_dir=cache_dir, dataset=dataset, debugging=debugging)
        print("Cache1 directory: ", self._dataset.cache_dir)
        self._dataset.download_and_prepare(download_config=download_config)
        self._dataset = self._dataset.as_dataset()
        self.sequence_lengths = self.seq_length()

    @property
    def dataset(self):
        return self._dataset

    def seq_length(self):
        train = [len(a["tokens"]) for a in self.dataset["train"]]
        dev = [len(a["tokens"]) for a in self.dataset["validation"]]
        test = [len(a["tokens"]) for a in self.dataset["test"]]
        return train + dev + test

    @property
    def labels(self) -> ClassLabel:
        return self._dataset['train'].features['ner_tags'].feature.names

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']

    def train_(self):
        return self._dataset['train_']

    def test(self):
        return self._dataset["test"]

    def test_(self):
        return self._dataset["test_"]

    def test_shuffled(self):
        return self._dataset["shuffled_test"]

    def validation(self):
        return self._dataset["dev"]

    def validation_(self):
        return self._dataset["dev_"]

    def data(self):
        return self._dataset["all_"]


if __name__ == '__main__':
    dataset = NERDataset().dataset

    print(dataset['train'])
    print(dataset['test'])
    print(dataset['validation'])

    print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)

    print("First sample: ", dataset['train'][80])

    lengths = [len(a["tokens"]) for a in dataset["train"]]
    import matplotlib

    matplotlib.use("TkAgg")
    plt.hist(lengths)
    plt.show()
    print("")
