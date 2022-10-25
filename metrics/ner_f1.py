#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: ner_f1.py
#
import wandb
from datasets import load_metric
import numpy as np
import pandas as pd

from plots.plot import plot_pos_dist

metric = load_metric("seqeval")



def compute_ner_f1(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results


def find_class_pos(labels, class_label):
    inds = []
    for i in range(len(labels)):
        if labels[i] in [f"B-{class_label}",f"I-{class_label}",f"E-{class_label}",f"S-{class_label}"]:
            inds = i,
    return inds


def compute_ner_pos_f1(p, label_list):
    predictions_scores, labels = p
    predictions = np.argmax(predictions_scores, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_scores = [
        [score for (score, l) in zip(scores, label) if l != -100]
        for scores, label in zip(predictions_scores, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    keys = list(results.keys())

    pos_dist = []
    for l in keys:
        if isinstance(results[l], dict):
            positions = []
            for sample in true_labels:
                positions += find_class_pos(sample, l)
            pos_dist += [(x,l,str(results[l]["f1"]*100)[:6]) for x in positions]
            # data_dict = {'positions': pd.Series(positions)}
            table = wandb.Table(data=[[a] for a in positions], columns=["positions"])
            results[l].update({"positions_distribution": table})
    position_dist = pd.DataFrame(pos_dist, columns=["position", "class","f1"])
    f = plot_pos_dist(position_dist)

    # "pr": wandb.plot.pr_curve(true_labels, true_scores, labels=label_list),
    # "roc": wandb.plot.roc_curve(true_labels, true_scores, labels=label_list),
    results.update({
                    "pos_dist": f})
    return results
