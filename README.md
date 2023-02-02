# P-22-NER-Position-Bias

This repository contains code for [Impact of Positional Bias on Language Models in Token Classification]().

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

## Bert Positional Bias in Named Entity Recognition and Part of Speech taggin
In this repository, we analyze BERT and other performance on two datasets [conll03](https://www.clips.uantwerpen.be/conll2003/ner/) and [Ontonotesv5](https://catalog.ldc.upenn.edu/LDC2013T19) for NER and [UD English Web TreeBank](https://github.com/UniversalDependencies/UD_English-EWT) and [TweeBank](https://github.com/Oneplus/Tweebank).

### Experiments:
Under the folder `experiments/scripts`,
#### 1. Position Bias Analysis

- Encoder Models: BERT, ERNIE, ELECTRA

        ./run_all_evaluations.sh
 
- Decoder Models: GPT2 and BLOOM

        ./run_all_evaluations_decoders.sh

#### 2. Fine-tuning with Random Position Perturbation, Context Perturbation

        ./run_all_finetuning.sh

### Results
All evaluation results and log files are synced to the Weights and Biases and will be published.
