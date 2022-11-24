from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def cosine_similarity(word_embeds: torch.FloatTensor,
                      position_embeds: torch.FloatTensor,
                      embedding_output: torch.FloatTensor,
                      attention_mask: torch.Tensor,
                      all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
                      all_self_attentions: Optional[Tuple[torch.FloatTensor]] = None,
                      k: Optional[torch.Tensor] = None):
    """
    Calculate cosine similarity between position and word embeddings, and between attentions score/probabilities.
    """
    cosine_positions = []
    cosine_words = []

    last_layer_position_similarity = F.cosine_similarity(position_embeds, embedding_output, dim=-1)

    attention_mask = attention_mask.float()

    last_layer_position_similarity = torch.mul(last_layer_position_similarity, attention_mask)
    cosine_positions.append(last_layer_position_similarity)

    last_layer_word_similarity = F.cosine_similarity(word_embeds, embedding_output, dim=-1)
    last_layer_word_similarity = torch.mul(last_layer_word_similarity, attention_mask)
    cosine_words.append(last_layer_word_similarity)

    if all_hidden_states is not None:
        for layer in all_hidden_states:
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

    output = {'position_cosine': cosine_positions, 'word_cosine': cosine_words}

    return output
