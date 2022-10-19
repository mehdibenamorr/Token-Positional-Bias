#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: train.py
#
from utils import set_random_seed

set_random_seed(23456)

import argparse
from typing import Dict, Union, Any, Optional, List
import os
from models.config import BertForTokenClassificationConfig
from models.bert_ner import BertForTokenClassification
from utils import get_parser
from dataset.ner_dataset import NERDataset
from dataset.ner_processor import NERProcessor
from dataset.collate_fn import DataCollator
import torch
import torch.nn as nn
from promise.dataloader import DataLoader
from metrics.pos_loss import CrossEntropyLossPerPosition
from transformers import Trainer, TrainingArguments
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"


class BertForNERTask(Trainer):
    def __init__(self, training_args: TrainingArguments, all_args: argparse.Namespace, **kwargs):
        # Args
        self.model_path = all_args.model
        self.dataset_name = all_args.dataset
        self.max_length = all_args.max_length
        self.pos_emb_type = all_args.position_embedding_type
        self.nbruns = all_args.nbruns
        self.preprocess = all_args.preprocess
        self.is_a_presaved_model = len(self.model_path.split('_')) > 1
        training_args.output_dir = os.path.join(str(Path.home()), training_args.output_dir)

        # Dataset
        self.dataset = NERDataset(dataset=self.dataset_name, debugging=all_args.debugging)
        self.processor = NERProcessor(pretrained_checkpoint=self.model_path)
        self.processed_dataset = self.dataset.dataset.map(self.processor.tokenize_and_align_labels, batched=True)
        self.collate_fn = DataCollator(tokenizer=self.processor.tokenizer)

        # Model loading
        bert_config = BertForTokenClassificationConfig.from_pretrained(self.model_path,
                                                                       id2label=self.dataset.id2label,
                                                                       label2id=self.dataset.label2id,
                                                                       position_embedding_type=self.pos_emb_type)
        print(f"DEBUG INFO -> check bert_config \n {bert_config}")
        model = BertForTokenClassification.from_pretrained(self.model_path, config=bert_config)

        super(BertForNERTask, self).__init__(model, args=training_args, train_dataset=self.processed_dataset["train"],
                                             eval_dataset=self.processed_dataset["validation"],
                                             data_collator=self.collate_fn, tokenizer=self.processor.tokenizer,
                                             **kwargs)
        self.test_dataset = self.processed_dataset["test"]
        self.loss_pos_fn = CrossEntropyLossPerPosition()
        self.losses = {"train": [], "dev": []}
        self.training = False

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        labels = inputs["labels"]
        logits = outputs["logits"]
        loss_per_pos = self.loss_pos_fn(logits, labels)
        if self.training:
            self.losses["train"].append(loss_per_pos)
        else:
            self.losses["dev"].append(loss_per_pos)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        self.training = True
        return super().training_step(model=model, inputs=inputs)

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ):

        self.training = False
        return super().evaluation_loop(dataloader=dataloader, description=description,
                                       prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys,
                                       metric_key_prefix=metric_key_prefix)

    def test(self):
        return super().evaluate(eval_dataset=self.test_dataset, metric_key_prefix="test")


def main():
    parser = get_parser()
    training_args, args = parser.parse_args_into_dataclasses()

    for i in range(args.nbruns):
        print(f"Run number:{i + 1}")
        task_trainer = BertForNERTask(all_args=args, training_args=training_args)
        task_trainer.train()

        ## Logging Loss per pos
        losses = task_trainer.losses

        task_trainer.test()


if __name__ == "__main__":
    main()
