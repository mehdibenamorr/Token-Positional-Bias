from transformers import AutoTokenizer


class NERProcessor(object):
    NAME = "NERProcessor"

    def __init__(self, pretrained_checkpoint, max_length=None, lower_case=True, kwargs=None):
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint, do_lower_case=lower_case)
        self.truncation_strategy = kwargs.truncation if kwargs else "do_not_truncate"
        self.return_truncated_tokens = kwargs.return_truncated_tokens if kwargs else False
        self.max_length = max_length

    @property
    def tokenizer(self):
        return self._tokenizer

    def tokenize_and_align_labels(self,
                                  examples,
                                  label_all_tokens=True, split="train"):

        def align_label(labels, word_ids, label_all_tokens=True):
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(labels[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx
            return label_ids

        if split == "train":
            tokenized_inputs = self._tokenizer(examples["tokens"],
                                               truncation=self.truncation_strategy,
                                               is_split_into_words=True,
                                               max_length=self.max_length,
                                               return_overflowing_tokens=self.return_truncated_tokens)
        else:
            tokenized_inputs = self._tokenizer(examples["tokens"],
                                               truncation=True,
                                               is_split_into_words=True)
        labels = []
        j = 0
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=j)
            overflowing_tokens = tokenized_inputs.encodings[j].overflowing if hasattr(tokenized_inputs.encodings[j],
                                                                                      "overflowing") else None
            label_ids = align_label(label, word_ids, label_all_tokens=label_all_tokens)
            labels.append(label_ids)
            if overflowing_tokens:
                for encoding in overflowing_tokens:
                    word_ids = encoding.word_ids
                    label_ids = align_label(label, word_ids, label_all_tokens)
                    labels.append(label_ids)
                    j += 1
            j += 1

        # Extract mapping between new and old indices
        sample_map = tokenized_inputs.pop("overflow_to_sample_mapping", None)
        if sample_map is not None:
            for key, values in examples.items():
                tokenized_inputs[key] = [values[i] for i in sample_map]
        tokenized_inputs["labels"] = labels
        return tokenized_inputs


if __name__ == '__main__':
    from dataset.ner_dataset import NERDataset, TruncateDataset

    checkpoint = "bert-base-uncased"
    conll03 = NERDataset(dataset="conll03")

    ner_processor = NERProcessor(pretrained_checkpoint=checkpoint)

    tokenized_datasets = conll03.dataset.map(ner_processor.tokenize_and_align_labels, batched=True)

    print(conll03)

    print("*" * 100)

    print(tokenized_datasets)

    print("First sample: ", conll03.dataset['train'][0])

    print("*" * 100)

    print("First tokenized sample: ", tokenized_datasets['train'][0])