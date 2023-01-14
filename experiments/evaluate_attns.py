#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: evaluate_attns.py
#
from utils import set_random_seed

set_random_seed(23456)
import argparse
from typing import Dict, Optional, List
import os
from models.config import ConfigForTokenClassification
from models.bert import BertForTokenClassification
from utils import get_parser
from dataset.ner_dataset import NERDataset
from dataset.ner_processor import NERProcessor
from dataset.collate_fn import DataCollator
import torch
from metrics.ner_f1 import ner_span_metrics, compute_ner_pos_f1
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import wandb
from transformers.trainer import logger
import inspect
import tempfile
from torch.utils.data import DataLoader, SequentialSampler

os.environ['WANDB_LOG_MODEL'] = "true"


# Fix for warning :The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# os.environ['WANDB_DISABLED'] = "true"


class TokenClassificationEvaluator(Trainer):
    def __init__(self, model_path: str, all_args: argparse.Namespace, dataset: NERDataset,
                 processor: NERProcessor, **kwargs):
        # Args
        self.model_path = model_path
        self.max_length = all_args.max_length
        self.watch_attentions = all_args.watch_attentions
        self.all_args = all_args
        self.is_a_presaved_model = len(self.model_path.split('_')) > 1

        training_args = TrainingArguments(
            self.model_path,
            per_device_eval_batch_size=all_args.batch_size,
            include_inputs_for_metrics=True,
            report_to=["wandb"]
        )
        self.dataset = dataset
        self.processor = processor
        self.collate_fn = DataCollator(tokenizer=processor.tokenizer, max_length=self.max_length,
                                       padding=self.all_args.padding)

        # Model loading
        bert_config = ConfigForTokenClassification.from_pretrained(self.model_path,
                                                                   watch_attentions=self.watch_attentions,
                                                                   output_attentions=self.watch_attentions,
                                                                   output_hidden_states=self.watch_attentions)
        print(f"DEBUG INFO -> check bert_config \n {bert_config}")
        model = BertForTokenClassification.from_pretrained(self.model_path, config=bert_config)

        super(TokenClassificationEvaluator, self).__init__(model, args=training_args,
                                                           data_collator=self.collate_fn, tokenizer=processor.tokenizer,
                                                           **kwargs)
        self.is_in_eval = True

    def test(self,
             test_dataset: Optional[Dataset] = None,
             ignore_keys: Optional[List[str]] = None,
             k: Optional[int] = None,
             duplicate_mode: Optional[str] = "none",
             metric_key_prefix: str = "test",
             ) -> Dict[str, float]:
        if k is None or duplicate_mode == "shift":
            self.compute_metrics = lambda p: compute_ner_pos_f1(p=p,
                                                                label_list=self.dataset.labels)
        else:
            self.compute_metrics = lambda p: ner_span_metrics(all_preds_scores=p.predictions,
                                                              all_labels=p.label_ids,
                                                              all_inputs=p.inputs,
                                                              label_list=self.dataset.labels,
                                                              k=k)
        return super().evaluate(eval_dataset=test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def eval_attn(self,
                  test_dataset: Optional[Dataset],
                  ):
        # dataloader = self.get_test_dataloader(test_dataset=test_dataset)
        test_sampler = SequentialSampler(test_dataset)
        dataloader = DataLoader(
            test_dataset,
            sampler=test_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        logger.info(f"***** Running Attention Evaluation *****")
        logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        batch_size = self.args.eval_batch_size
        logger.info(f"  Batch size = {batch_size}")
        sequence_info = []
        attention_scores = []
        raw_attention_scores = []
        positions_cosine = []
        words_cosine = []
        for step, inputs in enumerate(dataloader):
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.dissected_feed_forward)
            signature_columns = list(signature.parameters.keys())

            ignored_inputs = list(set(test_dataset.column_names) - set(signature_columns))
            fwd_inputs = {k: v for k, v in inputs.items() if k not in ignored_inputs}
            fwd_inputs = self._prepare_inputs(fwd_inputs)
            with torch.no_grad():
                outputs = self.model.dissected_feed_forward(**fwd_inputs, return_dict=False)
            cos_results = outputs[-2]
            attn_dict = outputs[-1]
            sequence_info.append({"id": inputs["id"][0], "tokens": inputs["original_tokens"][0],
                                  "labels": [self.dataset.id2label[l] for l in inputs["original_tags"][0]]})
            attention_scores.append(attn_dict["attention_probs"])
            raw_attention_scores.append(attn_dict["attention_scores"])
            positions_cosine.append(cos_results["positions_cosine"])
            words_cosine.append(cos_results["words_cosine"])
        tempdir = tempfile.TemporaryDirectory()
        seq_file = os.path.join(tempdir.name, "seq_info.pt")
        torch.save(sequence_info, seq_file)
        wandb.save(seq_file, policy="now")

        attn_scores = os.path.join(tempdir.name, "attention_scores.pt")
        torch.save(attention_scores, attn_scores)
        wandb.save(attn_scores, policy="now")

        attn_raw = os.path.join(tempdir.name, "attention_raw.pt")
        torch.save(raw_attention_scores, attn_raw)
        wandb.save(attn_raw, policy="now")

        position_cosine = os.path.join(tempdir.name, "pos_cos.pt")
        torch.save(positions_cosine, position_cosine)
        wandb.save(position_cosine, policy="now")

        word_cosine = os.path.join(tempdir.name, "word_cos.pt")
        torch.save(words_cosine, word_cosine)
        wandb.save(word_cosine, policy="now")
        return tempdir


def main():
    parser = get_parser(HF=False)
    args = parser.parse_args()
    experiment_name = f"{args.experiment}-{args.dataset}"
    entity = args.wandb_user
    os.environ["WANDB_DIR"] = args.wandb_dir
    api = wandb.Api()
    experiment_ref = f"{args.model.split('/')[-1]}-position_bias-{args.dataset}"
    runs = api.runs(entity + "/" + experiment_ref)
    tags = [f"max_length={args.max_length}", f"truncate={args.truncation}",
            f"padding={args.padding}",
            f"padding_side={args.padding_side}", f"pos_emb_type={args.position_embedding_type}",
            f"duplicate={args.duplicate}", f"log_attentions={args.watch_attentions}"]
    for i, run in enumerate(runs):
        config = vars(args)
        eval_run = wandb.init(project=experiment_name, name=run.name, tags=tags,
                              config=config)
        print(f"Run Name:{run.name}")

        # Dataset
        dataset = NERDataset(dataset=args.dataset, debugging=args.debugging)
        processor = NERProcessor(pretrained_checkpoint=args.model, max_length=args.max_length,
                                 kwargs=config)

        # Download the fine-tuned model
        run_path = "/".join(run.path[:-1])
        model_url = f"{run_path}/model-{run.id}:latest"
        model_artifact = eval_run.use_artifact(model_url, type="model")
        model_path = model_artifact.download()
        task_eval = TokenClassificationEvaluator(model_path, all_args=args, dataset=dataset, processor=processor)

        tempdir = None
        if args.duplicate:
            for k in range(1, 11):
                test_dataset = dataset.dataset["test_"].map(processor.tokenize_and_align_labels,
                                                            fn_kwargs={"duplicate": args.duplicate, "k": k,
                                                                       "duplicate_mode": args.duplicate_mode},
                                                            load_from_cache_file=False, batched=True)

                task_eval.test(test_dataset=test_dataset, metric_key_prefix=f"test_k={k}", k=k,
                               duplicate_mode=args.duplicate_mode)
        elif args.watch_attentions:
            test_dataset = dataset.dataset["test_"].map(processor.tokenize_and_align_labels,
                                                        fn_kwargs={"duplicate": True, "k": 10},
                                                        load_from_cache_file=False,
                                                        batched=True)
            tempdir = task_eval.eval_attn(test_dataset=test_dataset)

        wandb.finish()
        if tempdir is not None:
            tempdir.cleanup()
        task_eval = None
        import gc
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
