from .bert import BertForTokenClassification
from transformers import AutoModelForTokenClassification

model_mapping = {"bert-base-uncased": BertForTokenClassification,
                 "bert-large-uncased": BertForTokenClassification,
                 "google/electra-large-discriminator": AutoModelForTokenClassification,
                 "google/electra-base-discriminator": AutoModelForTokenClassification,
                 "nghuyong/ernie-2.0-base-en": AutoModelForTokenClassification,
                 "nghuyong/ernie-2.0-large-en": AutoModelForTokenClassification,
                 "zhiheng-huang/bert-base-uncased-embedding-relative-key-query": BertForTokenClassification,
                 "zhiheng-huang/bert-base-uncased-embedding-relative-key": BertForTokenClassification,
                 "bigscience/bloom-560m": AutoModelForTokenClassification,
                 "gpt2-large": AutoModelForTokenClassification,
                 "gpt2": AutoModelForTokenClassification,
                 }
