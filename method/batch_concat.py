def concatenate_batch(batch, max_length=512):
    print("here")
    input_ids = list(batch["input_ids"])
    attention_mask = list(batch["attention_mask"])
    token_type_ids = list(batch["token_type_ids"])
    labels = list(batch["labels"])

    # Get the original sequence
    original_input_idx = list(map(lambda x: x.nonzero().squeeze(), input_ids))
    seq_lengths = list(map(lambda x: x[x.nonzero()].squeeze().shape[0] - 1, input_ids))

    # Get subsets of sequence that add up to max_length

    # Concatenate the sequences and permutate in random orders

    return batch
