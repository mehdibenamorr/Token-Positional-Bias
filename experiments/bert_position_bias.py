#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: bert_position_bias.py
#
from torch.utils.data import DataLoader

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
from datasets import Dataset
import wandb

os.environ['WANDB_LOG_MODEL'] = "true"

# os.environ['WANDB_DISABLED'] = "true"


class BertForNERTask(Trainer):
    def __init__(self, training_args: TrainingArguments, all_args: argparse.Namespace, dataset: NERDataset,
                 train: Dataset, eval: Dataset,
                 processor: NERProcessor, **kwargs):
        # Args
        self.model_path = all_args.model
        self.max_length = all_args.max_length
        self.pos_emb_type = all_args.position_embedding_type
        self.nbruns = all_args.nbruns
        self.concatenate = all_args.concatenate
        self.all_args = all_args
        self.is_a_presaved_model = len(self.model_path.split('_')) > 1
        training_args.output_dir = os.path.join(str(Path.home()), training_args.output_dir)

        self.dataset = dataset
        self.processor = processor
        self.collate_fn = DataCollator(tokenizer=processor.tokenizer, max_length=self.max_length,
                                       padding=self.all_args.padding)

        # Model loading
        bert_config = BertForTokenClassificationConfig.from_pretrained(self.model_path,
                                                                       id2label=self.dataset.id2label,
                                                                       label2id=self.dataset.label2id,
                                                                       position_embedding_type=self.pos_emb_type)
        print(f"DEBUG INFO -> check bert_config \n {bert_config}")
        model = BertForTokenClassification.from_pretrained(self.model_path, config=bert_config)

        super(BertForNERTask, self).__init__(model, args=training_args, train_dataset=train,
                                             eval_dataset=eval,
                                             data_collator=self.collate_fn, tokenizer=processor.tokenizer,
                                             compute_metrics=lambda p: compute_ner_pos_f1(p=p,
                                                                                          label_list=self.dataset.labels),
                                             **kwargs)
        self.loss_pos_fn = CrossEntropyLossPerPosition()
        self.losses = {"train": [], "dev": []}
        self.is_in_eval = False

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        labels = inputs["labels"]
        logits = outputs["logits"]
        loss_per_pos = self.loss_pos_fn(logits, labels)
        if self.is_in_train and not self.is_in_eval:
            self.losses["train"].append(loss_per_pos)
        else:
            self.losses["dev"].append(loss_per_pos)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        return super().training_step(model=model, inputs=inputs)

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self.is_in_eval = True
        out = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                               metric_key_prefix=metric_key_prefix)
        self.is_in_eval = False
        return out

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ):

        eval_output = super().evaluation_loop(dataloader=dataloader, description=description,
                                              prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys,
                                              metric_key_prefix=metric_key_prefix)
        # Log position distribution
        metrics = eval_output.metrics
        for l in metrics.keys():
            if isinstance(metrics[l], dict):
                table = metrics[l].pop("positions_distribution", None)
                if table:
                    wandb.log({f'{l}.positions_distribution': wandb.plot.histogram(table, "positions",
                                                                                   title=f'{l}.positions_distribution')})
        pos_dist = metrics.pop(f"{metric_key_prefix}_pos_dist", None)
        if pos_dist is not None:
            wandb.log({
                f"{metric_key_prefix}_pos_dist": wandb.Image(pos_dist)})
        return EvalLoopOutput(predictions=eval_output.predictions, label_ids=eval_output.label_ids, metrics=metrics,
                              num_samples=eval_output.num_samples)

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

    def log_pos_losses(self):
        ## Logging Loss per pos
        train_losses = padded_stack(self.losses["train"]).view(-1,
                                                               self.max_length).detach().cpu().numpy()

        data = []
        for k in range(train_losses.shape[1]):
            data += [(loss, k) for loss in train_losses[:, k].tolist() if loss != 0]
        train_ = pd.DataFrame(data=data, columns=["loss", "position"])
        # table = wandb.Table(dataframe=train_) "train_loss/pos": table,
        f = plot_loss_dist(train_)
        wandb.log({"train_loss_dist": wandb.Image(f)})
        if self.losses["dev"]:
            dev_losses = padded_stack(self.losses["dev"]).view(-1,
                                                               self.max_length).detach().cpu().numpy()
            data = []
            for k in range(dev_losses.shape[1]):
                data += [(loss, k) for loss in train_losses[:, k].tolist() if loss != 0]
            eval_ = pd.DataFrame(data=data, columns=["loss", "position"])
            f = plot_loss_dist(eval_)
            # table = wandb.Table(dataframe=eval_) "eval_loss/pos": table,
            wandb.log({"eval_loss_dist": wandb.Image(f)})


def main():
    parser = get_parser()
    training_args, args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = training_args.output_dir
    experiment_name = f"{args.experiment}-{args.dataset}"
    tags = [f"max_length={args.max_length}", f"truncate={args.truncation}",
            f"padding={args.padding}", f"seed={training_args.seed}",
            f"padding_side={args.padding_side}", f"pos_emb_type={args.position_embedding_type}",
            f"duplicate={args.duplicate}"]
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

        task_trainer.log_pos_losses()

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


if __name__ == "__main__":
    main()
