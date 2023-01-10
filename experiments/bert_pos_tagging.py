#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: bert_pos_tagging.py
#

from utils import set_random_seed

set_random_seed(23456)
from experiments.bert_position_bias import BertForNERTask
import os
from utils import get_parser
from dataset.pos_dataset import POSDataset
from dataset.pos_processor import POSProcessor
import torch
import wandb

os.environ['WANDB_LOG_MODEL'] = "true"

# os.environ['WANDB_DISABLED'] = "true"


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
        dataset = POSDataset(dataset=args.dataset, debugging=args.debugging)
        processor = POSProcessor(pretrained_checkpoint=args.model, max_length=args.max_length,
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
