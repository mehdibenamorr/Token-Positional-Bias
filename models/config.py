#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: config.py
#
from transformers import BertConfig, ElectraConfig, ErnieConfig


class ConfigForTokenClassification(BertConfig, ElectraConfig, ErnieConfig):
    def __init__(self, **kwargs):
        super(ConfigForTokenClassification, self).__init__(**kwargs)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.0)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.truncated_normal = kwargs.get("truncated_normal", False)
        self.position_embedding_type = kwargs.get("position_embedding_type", "absolute")
        self.watch_attentions = kwargs.get("watch_attentions", False)
