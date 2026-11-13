"""Local adapters for EnsemJudge and Desklib classifier checkpoints.

EnsemJudge checkpoints supply classifier_hidden_size/classifier_hidden_dropout.
The built-in BERT forward provides pooler + dropout, followed by this MLP.
No Python file from a model repository is imported or executed.
"""

import torch
from torch import nn
from transformers import BertForSequenceClassification, DebertaV2Config, DebertaV2Model, PreTrainedModel, RobertaForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaPooler
from transformers.modeling_outputs import SequenceClassifierOutput


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


class LegacyRobertaPoolerClassifier(RobertaForSequenceClassification):
    """Load the legacy OpenAI snapshots' pooler without changing their CLS head.

    Modern RobertaForSequenceClassification omits this otherwise unused pooler.
    Retaining it accounts for every checkpoint key instead of allowing arbitrary
    unexpected weights; the author's classification head still consumes tokens.
    """

    def __init__(self, config):
        super().__init__(config)
        self.roberta.pooler = RobertaPooler(config)
        self.post_init()


class DesklibAIDetectionModel(PreTrainedModel):
    """Local masked-mean pooling and single-logit head from Desklib's model card."""

    config_class = DebertaV2Config
    base_model_prefix = "model"

    def __init__(self, config):
        # The published config omits num_labels; the checkpoint head has one row.
        config.num_labels = 1
        super().__init__(config)
        self.model = DebertaV2Model(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.model(input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids, return_dict=True)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(hidden).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return SequenceClassifierOutput(logits=self.classifier(pooled))


class CoCoDetModel(nn.Module):
    """Local ModernBERT + cosine head, matching the pinned author's infer.py.

    All three auxiliary heads are retained for strict checkpoint validation.
    The main labels are human/mix/ai; the cosine scale is the author's 30.0.
    No pretrained backbone weights or repository Python are needed at runtime.
    """

    def __init__(self, config):
        super().__init__()
        from transformers import ModernBertModel
        config.num_labels = 3
        config.reference_compile = False
        config._attn_implementation = "sdpa"
        self.config = config
        self.bert = ModernBertModel(config)
        self.main_classifier_weight = nn.Parameter(torch.empty(3, config.hidden_size))
        nn.init.xavier_uniform_(self.main_classifier_weight)
        self.content_classifier = nn.Linear(config.hidden_size, 8)
        self.style_classifier = nn.Linear(config.hidden_size, 7)
        self.fine_grained_classifier = nn.Linear(config.hidden_size, 6)

    @classmethod
    def from_local(cls, directory):
        from transformers import ModernBertConfig
        config = ModernBertConfig.from_pretrained(directory, local_files_only=True)
        model = cls(config)
        state = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
        # Unlike permissive state loading, this rejects missing auxiliary heads too.
        model.load_state_dict(state, strict=True)
        return model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        features = nn.functional.normalize(hidden, p=2, dim=1)
        weights = nn.functional.normalize(self.main_classifier_weight, p=2, dim=1)
        return SequenceClassifierOutput(logits=nn.functional.linear(features, weights) * 30.0)
