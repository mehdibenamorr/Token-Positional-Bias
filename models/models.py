from .bert import BertForTokenClassification
from .electra import ElectraForTokenClassification
from .ernie import ErnieForTokenClassification

model_mapping = {"bert-base-uncased": BertForTokenClassification,
                 "bert-base-cased": BertForTokenClassification,
                 "bert-large-uncased": BertForTokenClassification,
                 "bert-large-cased": BertForTokenClassification,
                 "google/electra-large-discriminator": ElectraForTokenClassification,
                 "google/electra-base-discriminator": ElectraForTokenClassification,
                 "nghuyong/ernie-2.0-base-en": ErnieForTokenClassification,
                 "nghuyong/ernie-2.0-large-en": ErnieForTokenClassification,
                 "zhiheng-huang/bert-base-uncased-embedding-relative-key-query": BertForTokenClassification,
                 "zhiheng-huang/bert-base-uncased-embedding-relative-key": BertForTokenClassification,
                 }
