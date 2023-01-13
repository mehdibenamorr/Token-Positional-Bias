#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: position_bias_cv.py
#


from utils import set_random_seed

set_random_seed(23456)
from experiments.position_bias import TokenClassificationTrainer
import os
from utils import get_parser
from dataset.ner_dataset import NERDataset
from dataset.ner_processor import NERProcessor
from sklearn.model_selection import KFold
import wandb
import torch

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
        # CV with 10-fold (k=10)
        dataset = NERDataset(dataset=args.dataset, debugging=args.debugging)
        processor = NERProcessor(pretrained_checkpoint=args.model, max_length=args.max_length,
                                 kwargs=config)

        all_data = dataset.dataset["all_"]

        kf = KFold(n_splits=args.cv)
        for cv, (train_idx, test_idx) in enumerate(kf.split(all_data)):
            print(f"CVFold={cv + 1}")
            # Run by cv
            wandb.init(project=experiment_name, name=f"CVFold={cv + 1}", tags=tags,
                       config=config)
            train_dataset = all_data.select(train_idx)
            train_eval_data = train_dataset.train_test_split(seed=training_args.seed, load_from_cache_file=False,
                                                             test_size=0.15)
            train_dataset = train_eval_data["train"].map(processor.tokenize_and_align_labels,
                                                         fn_kwargs={"concatenate": args.concatenate},
                                                         batched=True, load_from_cache_file=False)
            eval_dataset = train_eval_data["test"].map(processor.tokenize_and_align_labels, batched=True,
                                                       load_from_cache_file=False)
            test_dataset = all_data.select(test_idx)

            task_trainer = TokenClassificationTrainer(all_args=args, training_args=training_args, train=train_dataset,
                                                      eval=eval_dataset, dataset=dataset, processor=processor)
            task_trainer.train()

            # task_trainer.log_pos_losses()

            if args.duplicate:
                for k in range(1, 11):
                    test_dataset_ = test_dataset.map(processor.tokenize_and_align_labels,
                                                    fn_kwargs={"duplicate": args.duplicate, "k": k},
                                                    load_from_cache_file=False, batched=True)

                    task_trainer.test(test_dataset=test_dataset_, metric_key_prefix=f"test_k={k}", k=k)
            else:
                test_dataset_ = test_dataset.map(processor.tokenize_and_align_labels,
                                                fn_kwargs={"duplicate": True, "k": 1},
                                                load_from_cache_file=False,
                                                batched=True)
                task_trainer.test(test_dataset=test_dataset_, metric_key_prefix=f"test_k=1", k=1)

            wandb.finish()
            task_trainer = None
            import gc
            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
