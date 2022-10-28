#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: ner_f1.py
#
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""\
seqeval is a Python framework for sequence labeling evaluation.
seqeval can evaluate the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so on.

This is well-tested by using the Perl script conlleval, which can be used for
measuring the performance of a system that has processed the CoNLL-2000 shared task data.

seqeval supports following formats:
IOB1
IOB2
IOE1
IOE2
IOBES

See the [README.md] file at https://github.com/chakki-works/seqeval for more information.
"""
import importlib
from typing import Optional, List, Union

import datasets
import wandb
from datasets import load_metric
import numpy as np
import pandas as pd
from seqeval.metrics import classification_report, accuracy_score

from plots.plot import plot_pos_dist

# metric = load_metric("seqeval")


_CITATION = """\
@inproceedings{ramshaw-marcus-1995-text,
    title = "Text Chunking using Transformation-Based Learning",
    author = "Ramshaw, Lance  and
      Marcus, Mitch",
    booktitle = "Third Workshop on Very Large Corpora",
    year = "1995",
    url = "https://www.aclweb.org/anthology/W95-0107",
}
@misc{seqeval,
  title={{seqeval}: A Python framework for sequence labeling evaluation},
  url={https://github.com/chakki-works/seqeval},
  note={Software available from https://github.com/chakki-works/seqeval},
  author={Hiroki Nakayama},
  year={2018},
}
"""

_DESCRIPTION = """\
seqeval is a Python framework for sequence labeling evaluation.
seqeval can evaluate the performance of chunking tasks such as named-entity recognition, part-of-speech tagging, semantic role labeling and so on.

This is well-tested by using the Perl script conlleval, which can be used for
measuring the performance of a system that has processed the CoNLL-2000 shared task data.

seqeval supports following formats:
IOB1
IOB2
IOE1
IOE2
IOBES

See the [README.md] file at https://github.com/chakki-works/seqeval for more information.
"""


class Nereval(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/chakki-works/seqeval",
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="label"), id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/chakki-works/seqeval"],
            reference_urls=["https://github.com/chakki-works/seqeval"],
        )

    def _compute(
            self,
            predictions,
            references,
            suffix: bool = False,
            scheme: Optional[str] = None,
            mode: Optional[str] = None,
            sample_weight: Optional[List[int]] = None,
            zero_division: Union[str, int] = "warn",
    ):
        if scheme is not None:
            try:
                scheme_module = importlib.import_module("seqeval.scheme")
                scheme = getattr(scheme_module, scheme)
            except AttributeError:
                raise ValueError(f"Scheme should be one of [IOB1, IOB2, IOE1, IOE2, IOBES, BILOU], got {scheme}")
        report = classification_report(
            y_true=references,
            y_pred=predictions,
            suffix=suffix,
            output_dict=True,
            scheme=scheme,
            mode=mode,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        report.pop("macro avg")
        report.pop("weighted avg")
        overall_score = report.pop("micro avg")

        scores = {
            type_name: {
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1-score"],
                "number": score["support"],
            }
            for type_name, score in report.items()
        }
        scores["overall_precision"] = overall_score["precision"]
        scores["overall_recall"] = overall_score["recall"]
        scores["overall_f1"] = overall_score["f1-score"]
        scores["overall_accuracy"] = accuracy_score(y_true=references, y_pred=predictions)

        return scores


metric = Nereval()


def _find_class_pos(labels, class_label):
    inds = []
    for i in range(len(labels)):
        if labels[i] in [f"B-{class_label}", f"I-{class_label}", f"E-{class_label}", f"S-{class_label}"]:
            inds = i,
    return inds


def compute_ner_pos_f1(p, label_list):
    predictions_scores, labels, inputs = p
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
        np.vstack([score for (score, l) in zip(scores, label) if l != -100])
        for scores, label in zip(predictions_scores, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    keys = list(results.keys())

    pos_dist = []
    for l in keys:
        if isinstance(results[l], dict):
            positions = []
            for sample in true_labels:
                positions += _find_class_pos(sample, l)
            pos_dist += [(x, l, str(results[l]["f1"] * 100)[:6]) for x in positions]
            # data_dict = {'positions': pd.Series(positions)}
            table = wandb.Table(data=[[a] for a in positions], columns=["positions"])
            results[l].update({"positions_distribution": table})
    position_dist = pd.DataFrame(pos_dist, columns=["position", "class", "f1"])
    f = plot_pos_dist(position_dist)

    results.update({
        "pos_dist": f})

    return results


def ner_span_metrics(p, label_list, k):
    predictions_scores, labels, inputs = p
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
        np.vstack([score for (score, l) in zip(scores, label) if l != -100])
        for scores, label in zip(predictions_scores, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print("here")
