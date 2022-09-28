from torch.nn import CrossEntropyLoss
from transformers.trainer_callback import TrainerCallback
from transformers import Trainer
import torch
import wandb
evaluation_flag = False
losses_list = []


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
        if losses_list:
            loss_matrix = torch.stack(losses_list)
            wandb.log({"avg_loss_per_pos": loss_matrix.mean(dim=0).numpy(),
                       "std_loss_per_pos": loss_matrix.std(dim=0).numpy(),
                       "epoch": state.epoch})



class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        print("test")
        global evaluation_flag
        if evaluation_flag:
            global losses_list
            loss_fct = CrossEntropyLoss(reduction="none")
            outputs = model(**inputs)
            labels = inputs["labels"]
            logits = outputs["logits"]
            losses_list.append([loss_fct(logits[i], labels[i]) for i in range(logits.shape[0])])
        # ToDo use some weighted version of the loss function to improve training?
        return super().compute_loss(model, inputs, return_outputs=return_outputs)
