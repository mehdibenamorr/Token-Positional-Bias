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
                      attention_mask: torch.Tensor,
                      all_hidden_states: Optional[Dict] = None,
                      all_self_attentions: Optional[Dict] = None,
                      k: Optional[int] = None):
    """
    Calculate cosine similarity between position and word embeddings, and between attentions score/probabilities.
    """
    cosine_positions = []
    cosine_words = []

    # Chunk embeddings into k chunks
    chunk_size = int(embedding_output.shape[-2] / k)

    embedding_output = apply_chunking_to_tensors(chunk_size=chunk_size, chunk_dim=0, input_tensor=embedding_output)
    position_embeds = apply_chunking_to_tensors(chunk_size=chunk_size, chunk_dim=0, input_tensor=position_embeds)
    word_embeds = apply_chunking_to_tensors(chunk_size=chunk_size, chunk_dim=0, input_tensor=word_embeds)

    last_layer_position_similarity = F.cosine_similarity(position_embeds, embedding_output, dim=-1)

    attention_mask = attention_mask.float()

    last_layer_position_similarity = torch.mul(last_layer_position_similarity, attention_mask)
    cosine_positions.append(last_layer_position_similarity)

    last_layer_word_similarity = F.cosine_similarity(word_embeds, embedding_output, dim=-1)
    last_layer_word_similarity = torch.mul(last_layer_word_similarity, attention_mask)
    cosine_words.append(last_layer_word_similarity)

    if all_hidden_states is not None:
        for key, layer in all_hidden_states.items():
            layer_position_similarity = F.cosine_similarity(position_embeds, layer, dim=-1)
            layer_position_similarity = torch.mul(layer_position_similarity, attention_mask)
            cosine_positions.append(layer_position_similarity)

            layer_word_similarty = F.cosine_similarity(word_embeds, layer, dim=-1)
            layer_word_similarty = torch.mul(layer_word_similarty, attention_mask)
            cosine_words.append(layer_word_similarty)

    # cos : (batch, seq_len)
    # HERE
    cosine_positions = torch.stack(cosine_positions)
    cosine_positions = torch.transpose(cosine_positions, 0, 1)

    cosine_words = torch.stack(cosine_words)
    cosine_words = torch.transpose(cosine_words, 0, 1)

    output = {'positions_cosine': cosine_positions, 'words_cosine': cosine_words}

    return output
