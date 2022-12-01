from typing import Tuple, Optional, Dict

import torch
import torch.nn.functional as F


def apply_chunking_to_tensors(
        chunk_size: int, chunk_dim: int, input_tensor
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `cosine similarity` to each chunk independently to save memory.


    """

    if chunk_size > 0:
        tensor_shape = input_tensor.shape[chunk_dim]
        if input_tensor.shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {tensor_shape} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensor.shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensor_chunks = input_tensor.chunk(num_chunks, dim=chunk_dim)
        return input_tensor_chunks

    return input_tensor


def cosine_similarity(word_embeds: torch.FloatTensor,
                      position_embeds: torch.FloatTensor,
                      embedding_output: torch.FloatTensor,
                      all_hidden_states: Optional[Dict] = None,
                      k: Optional[int] = None):
    """
    Calculate cosine similarity between position and word embeddings, and between attentions score/probabilities.
    """
    # Chunk embeddings into k chunks
    chunk_size = int(embedding_output.shape[-2] / k)

    embedding_output = apply_chunking_to_tensors(chunk_size=chunk_size, chunk_dim=0, input_tensor=embedding_output)
    position_embeds = apply_chunking_to_tensors(chunk_size=chunk_size, chunk_dim=0, input_tensor=position_embeds)
    word_embeds = apply_chunking_to_tensors(chunk_size=chunk_size, chunk_dim=0, input_tensor=word_embeds)

    assert len(embedding_output) == len(position_embeds) == len(word_embeds)

    cosine_positions = {f"pos_sim_k={i + 1}": [] for i in range(k)}
    cosine_words = {f"word_sim_k={i + 1}": [] for i in range(k)}
    for i, embedding_at_k in enumerate(embedding_output):
        last_layer_position_similarity = F.cosine_similarity(position_embeds[0], embedding_at_k, dim=-1)
        last_layer_word_similarity = F.cosine_similarity(word_embeds[0], embedding_at_k, dim=-1)

        cosine_positions[f"pos_sim_k={i + 1}"].append(last_layer_position_similarity)
        cosine_words[f"word_sim_k={i + 1}"].append(last_layer_word_similarity)

    if all_hidden_states is not None:
        for key, layer in all_hidden_states.items():
            layer_emb = apply_chunking_to_tensors(chunk_size=chunk_size, chunk_dim=0, input_tensor=layer)
            for i, layer_emb_at_k in enumerate(layer_emb):
                layer_position_similarity = F.cosine_similarity(position_embeds[0], layer_emb_at_k, dim=-1)
                cosine_positions[f"pos_sim_k={i + 1}"].append(layer_position_similarity)

                layer_word_similarty = F.cosine_similarity(word_embeds[0], layer_emb_at_k, dim=-1)
                cosine_words[f"word_sim_k={i + 1}"].append(layer_word_similarty)

    # cosine metric : (seq_len, layers)
    cosine_positions = {k: torch.transpose(torch.stack(v), 0, 1) for k, v in cosine_positions.items()}
    cosine_words = {k: torch.transpose(torch.stack(v), 0, 1) for k, v in cosine_words.items()}

    output = {'positions_cosine': cosine_positions, 'words_cosine': cosine_words}

    return output
