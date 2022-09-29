import json
import os
import uuid

import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback

from .util import padded_stack, NpEncoder

evaluation_flag = False
losses_list = []
losses = []
BASE_PATH = os.path.join(os.path.expanduser('~'), ".pads-nlp")
MODELS_DIR = os.path.join(BASE_PATH, 'models', 'token_bias')


class LossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        global evaluation_flag
        evaluation_flag = True
        global losses_list
        losses_list = []

    def on_evaluate(self, args, state, control, **kwargs):
        global evaluation_flag
        evaluation_flag = False
        global losses_list
        global losses
        if losses_list:
            seq_length = max([a.shape[1] for a in losses_list])
            loss_matrix = padded_stack(losses_list[:-1]).view(-1, seq_length)
            losses.append({"avg_loss_per_pos": loss_matrix.mean(dim=0).cpu().numpy(),
                           "std_loss_per_pos": loss_matrix.std(dim=0).cpu().numpy(),
                           "epoch": state.epoch})
            with open(os.path.join(MODELS_DIR, 'results', f'losses_{uuid.uuid1()}.json'), 'w') as f:
                json.dump(losses, f, cls=NpEncoder)


class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        global evaluation_flag
        if evaluation_flag:
            global losses_list
            loss_fct = CrossEntropyLoss(reduction="none")
            outputs = model(**inputs)
            labels = inputs["labels"]
            logits = outputs["logits"]
            losses_list.append(torch.stack([loss_fct(logits[i], labels[i])[1:-1] for i in range(logits.shape[0])]))
        # ToDo use some weighted version of the loss function to improve training?
        return super().compute_loss(model, inputs, return_outputs=return_outputs)
