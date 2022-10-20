# P-22-NER-Position-Bias

This repository contains code for [Impact of Positional Bias in Data-imbalanced NER Downstream Task]().

## Requirements

    - datasets==2.5.1
    - tokenizers==0.12.1
    - transformers==4.22.1
    - torch==1.12.0
    - wandb==0.13.4
    - seaborn==0.12.1
    - seqeval==1.2.2


## Setup

* Install the conda environment
The environment used can reproduced by using `p-22-ner-position-bias.yml`
```shell
conda env create -f p-22-ner-position-bias.yml
```

## Bert Positional Bias in Named Entity Recognition
In this repository, we analyze BERT performance on two datasets [conll03]() and [Ontonotesv5](). Processed files for both datasets can be downloaded from [here]().

### Experiments:

#### 1. Bert Bias analysis with different Sequences lengths

For `max`:
- Run `experiments/bert_position_bias_ner.sh $dataset max`
For `median`:
- Run `experiments/bert_position_bias_ner.sh $dataset median`
For `min`:
- Run `experiments/bert_position_bias_ner.sh $dataset min`
 
#### 2. Fine-tuning with shuffling (validation), random padding,



### Results
The training loss and evaluation results on the dev set are synced to the wandb dashboard.
