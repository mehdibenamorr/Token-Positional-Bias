from transformers import AutoTokenizer



class NERProcessor(object):
    NAME = "NERProcessor"

    def __init__(self, pretrained_checkpoint, lower_case=True):
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint, do_lower_case=lower_case)

    @property
    def tokenizer(self):
        return self._tokenizer

    def tokenize_and_align_labels(self,
                                  examples,
                                  label_all_tokens=True):
        tokenized_inputs = self._tokenizer(examples["tokens"],
                                           truncation=True,
                                           is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

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
