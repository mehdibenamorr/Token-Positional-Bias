from datasets import load_metric
import numpy as np

metric = load_metric("seqeval")


def compute_ner_f1(predictions, labels, label_list):
    # predictions, labels = p
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
        if labels[i] == f"B-{class_label}" or labels[i] == f"I-{class_label}":
            inds += i,
    return inds


def compute_ner_pos_f1(p, label_list):
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
    nb_classes = int((len(label_list) - 1) / 2)
    classes = list(results.keys())[:nb_classes]

    for l in classes:
        positions = []
        for sample in true_labels:
            positions += find_class_pos(sample, l)
        results[l].update({"positions": positions})

    return results
