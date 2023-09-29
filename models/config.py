#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: config.py
#
from transformers import BertConfig, AutoConfig


class BertConfigForTokenClassification(BertConfig):
    def __init__(self, **kwargs):
        super(BertConfigForTokenClassification, self).__init__(**kwargs)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.0)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.truncated_normal = kwargs.get("truncated_normal", False)
        self.position_embedding_type = kwargs.get("position_embedding_type",
                                                  "absolute") if self.position_embedding_type == "absolute" else self.position_embedding_type
        self.watch_attentions = kwargs.get("watch_attentions", False)


config_mapping = {"bert-base-uncased": BertConfigForTokenClassification,
                  "bert-large-uncased": BertConfigForTokenClassification,
                  "google/electra-large-discriminator": AutoConfig,
                  "google/electra-base-discriminator": AutoConfig,
                  "nghuyong/ernie-2.0-base-en": AutoConfig,
                  "nghuyong/ernie-2.0-large-en": AutoConfig,
                  "zhiheng-huang/bert-base-uncased-embedding-relative-key-query": BertConfigForTokenClassification,
                  "zhiheng-huang/bert-base-uncased-embedding-relative-key": BertConfigForTokenClassification,
                  "bigscience/bloom-560m": AutoConfig,
                  "gpt2-large": AutoConfig,
                  "gpt2": AutoConfig,
                  "junnyu/roformer_chinese_base": AutoConfig,
                  }
