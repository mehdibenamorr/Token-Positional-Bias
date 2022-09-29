import argparse
import os

import datasets
import wandb
from transformers import TrainingArguments, set_seed

from src.train import train

PROJECT = "Position_Bias"
BASE_PATH = os.path.join(os.path.expanduser('~'), ".pads-nlp")
logger = datasets.logging.get_logger(__name__)
MODELS_DIR = os.path.join(BASE_PATH, 'models', 'token_bias')

parser = argparse.ArgumentParser(description='Token position bias in Bert Model')

# Mandatory
parser.add_argument('-model', type=str, default='bert-base-uncased',
                    help='Name of a specific model previously saved inside "models" folder'
                         ' or name of an HuggingFace model')
# parser.add_argument('dataset', type=str,
#                     help='Name of a specific Doccano dataset present inside "datasets" folder.')

# Optional
parser.add_argument('-run', type=str, default='conll2003',
                    help='Name of the run')
parser.add_argument('-nbruns', default=1, type=int, help='Number of epochs during training')
parser.add_argument('-shuffled', action='store_true', help='If set, training will be done on shuffled data')
parser.add_argument('-splitseed', default=42, type=int, help='Seed for reproducibility when sampling dataset. '
                                                             'Default is 42 (set None for randomness).')
parser.add_argument('-notrain', action='store_true', help='If set, training will be skipped')
parser.add_argument('-noeval', action='store_true', help='If set, evaluation phase will be skipped')
parser.add_argument('-plot', action='store_true', help='If set, charts will be plotted')
parser.add_argument('-maxseqlen', default=512, type=int,
                    help='Value used by tokenizer to apply padding or truncation to sequences. Default is '
                         'max_value=512')
parser.add_argument('-epochs', default=1, type=int, help='Number of epochs during training')
parser.add_argument('-warmsteps', default=500, type=int, help='Number of warm-up steps before training')
parser.add_argument('-lr', default=2e-5, type=float, help='Learning rate to use during training')
parser.add_argument('-wdecay', default=0.01, type=float, help='Weight decay to use during training')
parser.add_argument('-trainbatch', default=16, type=int, help='Per device batch size during training')
parser.add_argument('-evalbatch', default=1, type=int, help='Per device batch size during evaluation')
parser.add_argument('-logsteps', default=500, type=int, help='Number of training steps between 2 logs')
parser.add_argument('-savesteps', default=2000, type=int, help='Number of training steps between checkpoints saving')
parser.add_argument('-evalstrategy', default='steps', type=str, help='Strategy for evaluating model during training')

if __name__ == "__main__":
    cmd_args = parser.parse_args()
    os.environ['WANDB_PROJECT'] = PROJECT
    os.environ['WANDB_LOG_MODEL'] = "true"
    # Setting up directories according to the model_name provided in command line
    model_name_or_path = cmd_args.model
    is_a_presaved_model = len(model_name_or_path.split('_')) > 1
    set_seed(cmd_args.splitseed)
    run_name = cmd_args.run + "_" + model_name_or_path
    wandb.init(project=PROJECT,
               name=run_name,
               tags=["p-22-ner-position-bias"])
    model_output_dir = os.path.join(MODELS_DIR, model_name_or_path, run_name)

    model_cache_dir = os.path.join('cache', model_name_or_path)

    nb_runs = cmd_args.nbruns
    # If we are only evaluating a model then we save the results (and the logs) inside a specific folder
    model_eval_dir = os.path.join(*[model_output_dir, 'evaluations', 'position_bias'])

    model_logs_dir = os.path.join(model_eval_dir if cmd_args.notrain else model_output_dir, 'logs')

    print(f"Is a pre-saved model? {is_a_presaved_model}")
    print(f"Model Output dir: {model_output_dir}")
    print(f"Model Cache dir: {model_cache_dir}")
    print(f"Model Eval dir: {model_eval_dir}")
    print(f"Model Logs dir: {model_logs_dir}")

    hf_pretrained_model_checkpoint = cmd_args.model
    hf_pretrained_tokenizer_checkpoint = cmd_args.model

    args = TrainingArguments(
        model_output_dir,
        evaluation_strategy=cmd_args.evalstrategy,
        learning_rate=cmd_args.lr,
        warmup_steps=cmd_args.warmsteps,
        per_device_train_batch_size=cmd_args.trainbatch,
        per_device_eval_batch_size=cmd_args.evalbatch,
        logging_dir=model_logs_dir,
        logging_steps=cmd_args.logsteps,
        num_train_epochs=cmd_args.epochs,
        weight_decay=0.01,
        run_name=run_name,
    )

    dataset = datasets.load_dataset(cmd_args.run)
    MODELS_DIR = os.path.join(BASE_PATH, 'models', 'token_bias')
    for i in range(nb_runs):
        train(dataset, args, MODELS_DIR, model=hf_pretrained_model_checkpoint, )
