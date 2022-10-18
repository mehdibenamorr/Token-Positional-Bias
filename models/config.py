from transformers import BertConfig


class BertForTokenClassificationConfig(BertConfig):
    def __init__(self, **kwargs):
        super(BertForTokenClassificationConfig, self).__init__(**kwargs)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.0)
        self.num_labels = kwargs.get("num_labels", 2)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.truncated_normal = kwargs.get("truncated_normal", False)
        self.position_embedding_type = kwargs.get("position_embedding_type", "absolute")