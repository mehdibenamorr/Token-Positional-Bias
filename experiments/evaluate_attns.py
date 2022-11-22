#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: evaluate_attns.py
#


from utils import set_random_seed

set_random_seed(23456)
from plots.plot import plot_loss_dist
import pandas as pd
from transformers.trainer_utils import EvalLoopOutput
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
from metrics.pos_loss import CrossEntropyLossPerPosition, padded_stack
from metrics.ner_f1 import compute_ner_pos_f1, ner_span_metrics
from transformers import Trainer, TrainingArguments
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import Dataset
import wandb
import matplotlib.pyplot as plt

os.environ['WANDB_LOG_MODEL'] = "true"

os.environ['WANDB_DISABLED'] = "true"


class BertForNEREval(Trainer):
    def __init__(self, model_path: str, all_args: argparse.Namespace, dataset: NERDataset,
                 processor: NERProcessor, **kwargs):
        # Args
        self.model_path = model_path
        self.max_length = all_args.max_length
        self.log_attentions = all_args.log_attentions
        self.all_args = all_args
        self.is_a_presaved_model = len(self.model_path.split('_')) > 1

        training_args = TrainingArguments(
            self.model_path,
            do_predict=True,
            report_to=["none"],
            logging_strategy="no"
        )
        self.dataset = dataset
        self.processor = processor
        self.collate_fn = DataCollator(tokenizer=processor.tokenizer, max_length=self.max_length,
                                       padding=self.all_args.padding)

        # Model loading
        bert_config = BertForTokenClassificationConfig.from_pretrained(self.model_path,
                                                                       log_attentions=self.log_attentions)
        print(f"DEBUG INFO -> check bert_config \n {bert_config}")
        model = BertForTokenClassification.from_pretrained(self.model_path, config=bert_config)

        super(BertForNEREval, self).__init__(model, args=training_args,
                                             data_collator=self.collate_fn, tokenizer=processor.tokenizer,
                                             **kwargs)
        self.is_in_eval = True

    def test(self,
             test_dataset: Optional[Dataset] = None,
             ignore_keys: Optional[List[str]] = None,
             k: Optional[int] = None,
             metric_key_prefix: str = "test",
             ) -> Dict[str, float]:
        self.compute_metrics = lambda p: ner_span_metrics(all_preds_scores=p.predictions,
                                                          all_labels=p.label_ids,
                                                          all_inputs=p.inputs,
                                                          label_list=self.dataset.labels,
                                                          k=k)
        return super().evaluate(eval_dataset=test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ["WANDB_DIR"] = training_args.output_dir
    experiment_name = f"{args.experiment}-{args.dataset}"
    entity = args.wandb_user

    for i in range(args.nbruns):
        config = vars(args)
        wandb.init(project=experiment_name, name=f"run={i + 1}", tags=tags,
                   config=config)
        print(f"Run number:{i + 1}")

        # Dataset
        dataset = NERDataset(dataset=args.dataset, debugging=args.debugging)
        processor = NERProcessor(pretrained_checkpoint=args.model, max_length=args.max_length,
                                 kwargs=config)

        train_dataset = dataset.dataset["train_"].map(processor.tokenize_and_align_labels,
                                                      fn_kwargs={"concatenate": args.concatenate},
                                                      batched=True)
        eval_dataset = dataset.dataset["dev_"].map(processor.tokenize_and_align_labels, batched=True)

        task_trainer = BertForNERTask(all_args=args, training_args=training_args, train=train_dataset,
                                      eval=eval_dataset, dataset=dataset, processor=processor)
        task_trainer.train()
        #
        # task_trainer.evaluate()

        # task_trainer.log_pos_losses()

        if args.duplicate:
            for k in range(1, 11):
                test_dataset = dataset.dataset["test_"].map(processor.tokenize_and_align_labels,
                                                            fn_kwargs={"duplicate": args.duplicate, "k": k},
                                                            load_from_cache_file=False, batched=True)

                task_trainer.test(test_dataset=test_dataset, metric_key_prefix=f"test_k={k}", k=k)
        else:
            test_dataset = dataset.dataset["test_"].map(processor.tokenize_and_align_labels,
                                                        fn_kwargs={"duplicate": True, "k": 1},
                                                        load_from_cache_file=False,
                                                        batched=True)
            task_trainer.test(test_dataset=test_dataset, metric_key_prefix=f"test_k=10", k=1)

        wandb.finish()
        task_trainer = None
        import gc
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
