#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: config.py
#
from transformers import BertConfig, ElectraConfig, ErnieConfig


class BertConfigForTokenClassification(BertConfig):
    def __init__(self, **kwargs):
        super(BertConfigForTokenClassification, self).__init__(**kwargs)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.0)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.truncated_normal = kwargs.get("truncated_normal", False)
        self.position_embedding_type = kwargs.get("position_embedding_type",
                                                  "absolute") if self.position_embedding_type == "absolute" else self.position_embedding_type
        self.watch_attentions = kwargs.get("watch_attentions", False)


class ElectraConfigForTokenClassification(ElectraConfig):
    def __init__(self, **kwargs):
        super(ElectraConfigForTokenClassification, self).__init__(**kwargs)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.0)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.truncated_normal = kwargs.get("truncated_normal", False)
        self.position_embedding_type = kwargs.get("position_embedding_type",
                                                  "absolute") if self.position_embedding_type == "absolute" else self.position_embedding_type
        self.watch_attentions = kwargs.get("watch_attentions", False)


class ErnieConfigForTokenClassification(ErnieConfig):
    def __init__(self, **kwargs):
        super(ErnieConfigForTokenClassification, self).__init__(**kwargs)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.0)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.truncated_normal = kwargs.get("truncated_normal", False)
        self.position_embedding_type = kwargs.get("position_embedding_type",
                                                  "absolute") if self.position_embedding_type == "absolute" else self.position_embedding_type
        self.watch_attentions = kwargs.get("watch_attentions", False)


config_mapping = {"bert-base-uncased": BertConfigForTokenClassification,
                 "bert-base-cased": BertConfigForTokenClassification,
                 "bert-large-uncased": BertConfigForTokenClassification,
                 "bert-large-cased": BertConfigForTokenClassification,
                 "google/electra-large-discriminator": ElectraConfigForTokenClassification,
                 "google/electra-base-discriminator": ElectraConfigForTokenClassification,
                 "nghuyong/ernie-2.0-base-en": ErnieConfigForTokenClassification,
                 "nghuyong/ernie-2.0-large-en": ErnieConfigForTokenClassification,
                 "zhiheng-huang/bert-base-uncased-embedding-relative-key-query": BertConfigForTokenClassification,
                 "zhiheng-huang/bert-base-uncased-embedding-relative-key": BertConfigForTokenClassification,
                 }
