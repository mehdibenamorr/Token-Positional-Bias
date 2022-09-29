import json
import os
import uuid

import numpy as np
import wandb
from transformers import AutoModelForTokenClassification, Trainer
from transformers import DataCollatorForTokenClassification
from src.callback import CustomLossTrainer
from .metrics import compute_metrics, metric, find_class_pos
from .tokenizer import HFTokenizer
from .callback import LossCallback
from .util import NpEncoder

BASE_PATH = os.path.join(os.path.expanduser('~'), ".pads-nlp")
MODELS_DIR = os.path.join(BASE_PATH, 'models', 'token_bias')


def train(dataset, training_args, saving_folder, model="bert-base-uncased"):
    labels = dataset['train'].features['ner_tags'].feature.names
    hf_preprocessor = HFTokenizer.init_vf(hf_pretrained_tokenizer_checkpoint=model)
    model = AutoModelForTokenClassification.from_pretrained(model, num_labels=len(labels))
    # tokenizer = AutoTokenizer.from_pretrained(model)  # Todo tokenizer?

    tokenized_datasets = dataset.map(hf_preprocessor.tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=hf_preprocessor.tokenizer)

    saving_folder = f"{saving_folder}/{str(uuid.uuid1())}"

    trainer = CustomLossTrainer(
        model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=hf_preprocessor.tokenizer,
        compute_metrics=lambda p: compute_metrics(p=p, label_list=labels),
        callbacks=[LossCallback()]
    )

    trainer.train()
    trainer.evaluate()

    # Test Set evaluation ToDO
    # predictions = trainer.predict(tokenized_datasets["test"])
    # print(predictions.predictions.shape, predictions.label_ids.shape)
    # preds = np.argmax(predictions.predictions, axis=-1)
    #
    # metric_result = metric.compute(predictions=preds, references=predictions.label_ids)

    # Predictions on test dataset and evaluation

    predictions, labels_, _ = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels_)
    ]

    true_labels = [
        [labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels_)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    nb_classes = int((len(labels) - 1) / 2)
    classes = list(results.keys())[:nb_classes]

    for l in classes:
        positions = []
        for sample in true_labels:
            positions += find_class_pos(sample, l)
        results[l].update({"positions": positions})

    with open(os.path.join(MODELS_DIR, 'results', f'test_{uuid.uuid1()}.json'), 'w') as f:
        json.dump(results, f, cls=NpEncoder)
    test_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                test_results["test-{}".format(kk)] = vv
        else:
            test_results["test-{}".format(k)] = v
    print(results)
    wandb.log(test_results)
    trainer.save_model(saving_folder)

    return results
