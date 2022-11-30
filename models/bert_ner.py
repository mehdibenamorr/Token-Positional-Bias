#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: bert_ner.py
import math
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, apply_chunking_to_forward
from transformers.modeling_outputs import TokenClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

from metrics.cosine_similartiy import cosine_similarity


def clip_tensor(input_tensor: torch.Tensor, mask: torch.BoolTensor, dim=None):
    if dim is not None:
        return input_tensor


class BertForTokenClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.watch_attentions = config.watch_attentions
        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
        self.attn_dict = {}

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            k: Optional[List] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        k_factor = k[0][0]
        if self.config.watch_attentions:
            outputs = self.dissected_bert_forward(input_ids,
                                                  attention_mask=attention_mask,
                                                  token_type_ids=token_type_ids,
                                                  position_ids=position_ids,
                                                  head_mask=head_mask,
                                                  inputs_embeds=inputs_embeds,
                                                  output_attentions=output_attentions,
                                                  output_hidden_states=output_hidden_states,
                                                  return_dict=return_dict,
                                                  k=k_factor)
        else:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def dissected_bert_forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            k: int = None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.bert.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.bert.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.bert.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape)

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # Dissected Encoder code (Start)
        sequence_mask = attention_mask.squeeze(0) == 1
        sequence_mask[0] = sequence_mask[attention_mask.sum().item() - 1] = False  # [CLS] and [SEP] tokens
        hidden_states = embedding_output
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        attn_dict = {}
        embs_dict = {}
        for i, layer_module in enumerate(self.bert.encoder.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # Dissected BertLayer code (Start)

            # Dissected BertAttention code (Start)
            # Dissected SelfAttention code (Start)
            mixed_query_layer = layer_module.attention.self.query(hidden_states)
            query_layer = layer_module.attention.self.transpose_for_scores(mixed_query_layer)
            key_layer = layer_module.attention.self.transpose_for_scores(layer_module.attention.self.key(hidden_states))
            value_layer = layer_module.attention.self.transpose_for_scores(
                layer_module.attention.self.value(hidden_states))

            # Take the dot product between "query" and "key" to get the raw attention scores.
            self_attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            self_attention_scores = self_attention_scores / math.sqrt(layer_module.attention.self.attention_head_size)
            if extended_attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                self_attention_scores = self_attention_scores + extended_attention_mask

            # Normalize the attention scores to probabilities.
            self_attention_probs = nn.functional.softmax(self_attention_scores, dim=-1)

            attention_scores = self_attention_scores.squeeze(0)[:, sequence_mask, :][:, :, sequence_mask]
            attention_probs = self_attention_probs.squeeze(0)[:, sequence_mask, :][:, :, sequence_mask]

            attn_dict.update({f"layer_attn_{i}": {"attention_scores": attention_scores,
                                                  "attention_probs": attention_probs}})

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            self_attention_probs = layer_module.attention.self.dropout(self_attention_probs)

            # Mask heads if we want to
            if layer_head_mask is not None:
                self_attention_probs = self_attention_probs * layer_head_mask

            context_layer = torch.matmul(self_attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (layer_module.attention.self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            self_attention_outputs = (context_layer, self_attention_probs)
            # Dissected SelfAttention code (End)

            attention_output = layer_module.attention.output(self_attention_outputs[0], hidden_states)

            attention_outputs = (attention_output,) + self_attention_outputs[1:]

            # Dissected BertAttention code (End)
            layer_output = apply_chunking_to_forward(
                layer_module.feed_forward_chunk, layer_module.chunk_size_feed_forward, layer_module.seq_len_dim,
                attention_output
            )

            layer_outputs = (layer_output,) + attention_outputs[1:]
            # Dissected BertLayer code (End)

            hidden_states = layer_outputs[0]
            embs_dict.update({f"layer_emb_{i}": hidden_states[:, sequence_mask, :].squeeze(0)})
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Dissected Encoder code (End)
        sequence_output = hidden_states
        pooled_output = self.bert.pooler(sequence_output) if self.bert.pooler is not None else None

        # First calculate cosine simlarity with intermediate representations wih each position (per k)
        embedding_output_cut = embedding_output.squeeze(0)[sequence_mask, :]
        inputs = input_ids[:, sequence_mask].squeeze(0)
        seq_len = embedding_output_cut.shape[-2]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=embedding_output.device)

        position_embs = self.bert.embeddings.position_embeddings(position_ids)
        word_embs = self.bert.embeddings.word_embeddings(inputs)

        cos_results = cosine_similarity(word_embeds=word_embs, position_embeds=position_embs,
                                        embedding_output=embedding_output_cut, attention_mask=attention_mask,
                                        all_hidden_states=embs_dict, all_self_attentions=attn_dict,
                                        k=k)

        if not return_dict:
            return tuple(
                v
                for v in [
                    sequence_output,
                    pooled_output,
                    all_hidden_states,
                    all_self_attentions,
                    cos_results,
                    attn_dict
                ]
                if v is not None
            )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
