"""Local implementation of the BERT checkpoints' two-layer classifier.

EnsemJudge checkpoints supply classifier_hidden_size/classifier_hidden_dropout.
The built-in BERT forward provides pooler + dropout, followed by this MLP.
No Python file from a model repository is imported or executed.
"""

from torch import nn
from transformers import BertForSequenceClassification


class BertMlpClassifier(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        hidden = config.classifier_hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, hidden),
            nn.ReLU(),
            nn.Dropout(config.classifier_hidden_dropout),
            nn.Linear(hidden, config.num_labels),
        )
        self.post_init()
