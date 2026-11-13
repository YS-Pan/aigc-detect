"""Windows retain the classifier's pre-tokenized IDs across tokenizer APIs."""
import pytest

from aigc_detector.engine import ModelUnavailable, prepare_window


@pytest.mark.parametrize("model_id,names,expected", [
    ("mpu-zh-v3", ["input_ids", "token_type_ids", "attention_mask"],
     {"input_ids": [101, 17, 18, 102], "token_type_ids": [0, 0, 0, 0], "attention_mask": [1, 1, 1, 1]}),
    ("hc3-en", ["input_ids", "attention_mask"],
     {"input_ids": [101, 17, 18, 102], "attention_mask": [1, 1, 1, 1]}),
])
def test_fast_tokenizer_without_prepare_for_model(model_id, names, expected):
    class Tokenizer:
        cls_token_id = 101
        sep_token_id = 102
        model_input_names = names
        def num_special_tokens_to_add(self, pair=False):
            assert pair is False
            return 2
    assert prepare_window(Tokenizer(), [17, 18], model_id) == expected


def test_unrecognized_fast_tokenizer_template_fails_instead_of_changing_scores():
    class Tokenizer:
        cls_token_id = 101
        sep_token_id = 102
        def num_special_tokens_to_add(self, pair=False):
            return 3
    with pytest.raises(ModelUnavailable, match="Unsupported single-sequence tokenizer template"):
        prepare_window(Tokenizer(), [17, 18], "mpu-zh-v3")


def test_older_tokenizers_keep_their_existing_prepare_for_model():
    class Tokenizer:
        def prepare_for_model(self, ids, **kwargs):
            assert kwargs == {"add_special_tokens": True, "return_attention_mask": True,
                              "truncation": False, "verbose": False}
            return {"input_ids": [91, *ids, 92], "attention_mask": [1] * (len(ids) + 2)}
    assert prepare_window(Tokenizer(), [7], "mpu-zh-v3")["input_ids"] == [91, 7, 92]
