#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: position_bias.py
#
from utils import set_random_seed

set_random_seed(23456)
from plot_utils.plot import plot_loss_dist
import pandas as pd
import argparse
from typing import Dict, Union, Any, Optional, List
import os
from models.config import config_mapping
from models import model_mapping
from utils import get_parser
from dataset.ner_dataset import NERDataset
from dataset.ner_processor import NERProcessor
from dataset.pos_dataset import POSDataset
from dataset.pos_processor import POSProcessor
from dataset.collate_fn import DataCollator
import torch
import torch.nn as nn
from metrics.pos_loss import CrossEntropyLossPerPosition, padded_stack
from metrics.ner_f1 import compute_ner_pos_f1, ner_span_metrics
from transformers import Trainer, TrainingArguments, is_apex_available
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import Dataset
import wandb
import matplotlib.pyplot as plt

from transformers.utils import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

from method.batch_concat import concatenate_batch
from method.position_shift import random_shift

if is_apex_available():
    from apex import amp

os.environ['WANDB_LOG_MODEL'] = "true"

os.environ['WANDB_DISABLED'] = "true"


class TokenClassificationTrainer(Trainer):
    def __init__(self, training_args: TrainingArguments, all_args: argparse.Namespace,
                 dataset: Union[NERDataset, POSDataset],
                 train: Dataset, eval: Dataset,
                 processor: Union[NERProcessor, POSProcessor], **kwargs):
        # Args
        self.model_path = all_args.model
        self.max_length = all_args.max_length
        self.nbruns = all_args.nbruns
        self.concatenate = all_args.concatenate
        self.position_shift = all_args.position_shift
        self.all_args = all_args
        self.is_a_presaved_model = len(self.model_path.split('_')) > 1
        training_args.output_dir = os.path.join(str(Path.home()), training_args.output_dir)

        self.dataset = dataset
        self.processor = processor
        self.collate_fn = DataCollator(tokenizer=processor.tokenizer, max_length=self.max_length,
                                       padding=self.all_args.padding)

        # Model loading
        config_cls = config_mapping[self.model_path]
        model_config = config_cls.from_pretrained(self.model_path,
                                                  id2label=self.dataset.id2label,
                                                  label2id=self.dataset.label2id)
        print(f"DEBUG INFO -> check bert_config \n {model_config}")
        if self.model_path in model_mapping:
            model_cls = model_mapping[self.model_path]
        else:
            raise ValueError(f"Model {self.model_path} not supported")
        model = model_cls.from_pretrained(self.model_path, config=model_config)

        super(TokenClassificationTrainer, self).__init__(model, args=training_args, train_dataset=train,
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
        """
        Perform a training step on a batch of inputs.

        Two processing methods can be applied on each batch of inputs, i.e. random-shift of position ids and
        concatenation with random premutations of position ids.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        hf_loss = super().training_step(model, inputs)
        # Rerun the training step on the augmented batch
        _inputs = None
        if self.concatenate:
            _inputs = concatenate_batch(inputs, max_length=self.max_length)
        elif self.position_shift:
            _inputs = random_shift(inputs, max_length=self.max_length)
        if _inputs is not None:
            _loss = super().training_step(model, _inputs)
            hf_loss += _loss
            return hf_loss
        return hf_loss

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
        return eval_output

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
        plt.close(f)
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
            plt.close(f)


def main():
    parser = get_parser()
    training_args, args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = training_args.output_dir
    experiment_name = f"{args.model.split('/')[-1]}-{args.experiment}-{args.dataset}"
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
        if args.dataset in ["conll03", "ontonotes5"]:
            dataset = NERDataset(dataset=args.dataset, debugging=args.debugging)
            processor = NERProcessor(pretrained_checkpoint=args.model, max_length=args.max_length,
                                     kwargs=config)
        elif args.dataset in ["en_ewt", "tweebank"]:
            dataset = POSDataset(dataset=args.dataset, debugging=args.debugging)
            processor = POSProcessor(pretrained_checkpoint=args.model, max_length=args.max_length,
                                     kwargs=config)
        else:
            raise ValueError(f"Dataset {args.dataset} not supported")

        train_dataset = dataset.dataset["train_"].map(processor.tokenize_and_align_labels,
                                                      batched=True)
        eval_dataset = dataset.dataset["dev_"].map(processor.tokenize_and_align_labels, batched=True)

        task_trainer = TokenClassificationTrainer(all_args=args, training_args=training_args, train=train_dataset,
                                                  eval=eval_dataset, dataset=dataset, processor=processor)
        task_trainer.train()

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
