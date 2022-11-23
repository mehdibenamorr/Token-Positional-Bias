import torch
import torch.nn.functional as F


def cosine_similarity(word_embs, position_embs, embedding_output, all_hidden_states, all_self_attentions,
                      attention_mask):
    cos_pos = []
    cos_word = []

    sims_pos = F.cosine_similarity(position_embs, embedding_output, dim=-1)

    attention_mask = attention_mask.float()

    sims_pos = torch.mul(sims_pos, attention_mask)
    cos_pos.append(sims_pos)

    sims_word = F.cosine_similarity(word_embs, embedding_output, dim=-1)
    sims_word = torch.mul(sims_word, attention_mask)
    cos_word.append(sims_word)

    for layer in all_hidden_states:
        sims_pos = F.cosine_similarity(position_embs, layer, dim=-1)
        sims_pos = torch.mul(sims_pos, attention_mask)
        cos_pos.append(sims_pos)

        sims_word = F.cosine_similarity(word_embs, layer, dim=-1)
        sims_word = torch.mul(sims_word, attention_mask)
        cos_word.append(sims_word)

    # cos : (batch, seq_len)
    # HERE
    cos_pos = torch.stack(cos_pos)
    cos_pos = torch.transpose(cos_pos, 0, 1)

    cos_word = torch.stack(cos_word)
    cos_word = torch.transpose(cos_word, 0, 1)

    cos = {'position': cos_pos, 'word': cos_word}

    return cos
