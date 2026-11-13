"""Explicit label semantics, verified against the authors' inference code/config."""

from pathlib import Path

# Development and historical evaluation scripts use this source-tree anchor.
# Installed runtime code does not write or find resources through it.
ROOT = Path(__file__).resolve().parents[1]

CATALOG = {
    "mpu-zh-v3": {
        "repo_id": "yuchuantian/AIGC_detector_zhv3",
        "name": "MPU Chinese v3",
        "ai_label_index": 1,
        "language": "zh",
        "label_source": "https://huggingface.co/spaces/yuchuantian/AIGC_text_detector/blob/191c982/app.py",
    },
    "mpu-zh-v3-short": {
        "repo_id": "yuchuantian/AIGC_detector_zhv3short",
        "name": "MPU Chinese v3 Short",
        "ai_label_index": 1,
        "language": "zh",
        "label_source": "https://huggingface.co/spaces/yuchuantian/AIGC_text_detector/blob/191c982/app.py",
    },
    "mpu-zh-v2": {
        "repo_id": "yuchuantian/AIGC_detector_zhv2",
        "name": "MPU Chinese v2",
        "ai_label_index": 1,
        "language": "zh",
        "label_source": "https://huggingface.co/spaces/yuchuantian/AIGC_text_detector/blob/191c982/app.py",
    },
    "hc3-zh": {
        "repo_id": "Hello-SimpleAI/chatgpt-detector-roberta-chinese",
        "name": "HC3 Chinese RoBERTa",
        "ai_label_index": 1,
        "language": "zh",
        "label_source": "https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta-chinese/blob/2f0f5f2af59af169a3f236c14ffad0fa7d6f072b/config.json",
    },
    "ensemjudge-bert": {
        "repo_id": "johnsonwangzs/mgtd-sys-model-weights",
        "revision": "1fa2938e03ea9d996360ed5b3050008a0e76aa8f",
        "subfolder": "chinese-bert-sft_03-28_20-17_908",
        "name": "EnsemJudge Chinese BERT",
        "loader": "bert_mlp",
        "tokenizer_repo_id": "hfl/chinese-bert-wwm-ext",
        "tokenizer_revision": "2a995a880017c60e4683869e817130d8af548486",
        "ai_label_index": 1,
        "language": "zh",
        "label_source": "https://github.com/NLP2CT/NLPCC-2025-Task1#data-format",
    },
    "ensemjudge-roberta": {
        "repo_id": "johnsonwangzs/mgtd-sys-model-weights",
        "revision": "1fa2938e03ea9d996360ed5b3050008a0e76aa8f",
        "subfolder": "chinese-roberta-sft_03-21_16-39_938",
        "name": "EnsemJudge Chinese RoBERTa Large",
        "loader": "bert_mlp",
        "tokenizer_repo_id": "hfl/chinese-roberta-wwm-ext-large",
        "tokenizer_revision": "a25cc9e05974bd9687e528edd516f2cfdb3f5db9",
        "ai_label_index": 1,
        "language": "zh",
        "label_source": "https://github.com/NLP2CT/NLPCC-2025-Task1#data-format",
    },
    "anx-bert-zh": {
        "repo_id": "AnxForever/chinese-ai-detector-bert",
        "revision": "346d23cbb1418790dc3d389faf54de665736516c",
        "name": "AnxForever Chinese BERT v11c",
        "ai_label_index": 1,
        "language": "zh",
        "label_source": "https://huggingface.co/AnxForever/chinese-ai-detector-bert/blob/346d23cbb1418790dc3d389faf54de665736516c/config.json",
    },
    "paperpass-v3": {
        "repo_id": "yibo365/paperpass-v3",
        "revision": "f042809a41e96208c25e69dbedf7a5c312a590c2",
        "name": "yibo365 Paperpass v3 (community)",
        "ai_label_index": 1,
        "language": "zh",
        "label_source": "https://huggingface.co/yibo365/paperpass-v3/blob/f042809a41e96208c25e69dbedf7a5c312a590c2/README.md",
    },
    "cmj-zh": {
        "repo_id": "mokawa3018/cmj-chinese-aigc-text-detector",
        "revision": "ade5833a8eae59444f6a3bb85b72417ca02da07f",
        "name": "CMJ Chinese AIGC detector",
        "ai_label_index": 1,
        "language": "zh",
        "label_source": "https://huggingface.co/mokawa3018/cmj-chinese-aigc-text-detector/blob/ade5833a8eae59444f6a3bb85b72417ca02da07f/config.json",
    },
    "bibbbu-xlm-roberta": {
        "repo_id": "bibbbu/multilingual-ai-human-detector_xlm-roberta-base",
        "revision": "b4aa2e06d8428b4d27f2ef625267044d74b2eeb8",
        "name": "bibbbu Multilingual XLM-RoBERTa Base",
        "ai_label_index": 1,
        "language": "zh,en,vi",
        "label_source": "https://github.com/vutuongvy101/multilingual-ai-human-text-detection/blob/1fbb02a7d77e454d4c3a21e8bcc262bd03db3190/src/multilingual_ai_detection/data/data_splitter.py#L59",
    },
    "hc3-en": {
        "repo_id": "Hello-SimpleAI/chatgpt-detector-roberta",
        "revision": "d2b342c61775d5dd0221808a79983ed3b86ffd86",
        "name": "HC3 English RoBERTa",
        "ai_label_index": 1,
        "language": "en",
        "label_source": "https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta/blob/d2b342c61775d5dd0221808a79983ed3b86ffd86/config.json",
    },
    "desklib-en": {
        "repo_id": "desklib/ai-text-detector-v1.01",
        "revision": "5fdea974cd4287c61674951ec78803aa274e2fb7",
        "name": "Desklib English DeBERTa v1.01",
        "loader": "desklib_mean_pool",
        "score_activation": "sigmoid",
        "ai_label_index": 0,
        "language": "en",
        "label_source": "https://huggingface.co/desklib/ai-text-detector-v1.01/blob/5fdea974cd4287c61674951ec78803aa274e2fb7/README.md",
    },
    "fakespot-en": {
        "repo_id": "fakespot-ai/roberta-base-ai-text-detection-v1",
        "revision": "f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db",
        "name": "Fakespot English RoBERTa v1",
        "ai_label_index": 1,
        "language": "en",
        "label_source": "https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db/config.json",
    },
    "openai-base-en": {
        "repo_id": "openai-community/roberta-base-openai-detector",
        "revision": "6cba99c003b711c7fe94f8a3aa2be35a792cb6fa",
        "name": "OpenAI GPT-2 output detector RoBERTa Base",
        "loader": "legacy_roberta_pooler",
        "ai_label_index": 0, "language": "en",
        "label_source": "https://huggingface.co/openai-community/roberta-base-openai-detector/blob/6cba99c003b711c7fe94f8a3aa2be35a792cb6fa/config.json",
    },
    "openai-large-en": {
        "repo_id": "openai-community/roberta-large-openai-detector",
        "revision": "38f3e0ccf205e9c4e0b38ae9d75ec948141bf832",
        "name": "OpenAI GPT-2 output detector RoBERTa Large",
        "loader": "legacy_roberta_pooler",
        "ai_label_index": 0, "language": "en",
        "label_source": "https://github.com/openai/gpt-2-output-dataset/blob/b76f67c651fea691b37abe01a6de762bc43330e7/detector/server.py",
    },
    "piratexx-en": {
        "repo_id": "PirateXX/AI-Content-Detector",
        "revision": "0d5450c5baf7c6ccab387a120bcc9c2a4d2d3d9c",
        "name": "PirateXX AI Content Detector RoBERTa Base",
        "ai_label_index": 0, "language": "en",
        "label_source": "https://huggingface.co/PirateXX/AI-Content-Detector/blob/0d5450c5baf7c6ccab387a120bcc9c2a4d2d3d9c/README.md",
    },
    "mage-en": {
        "repo_id": "yaful/MAGE",
        "revision": "0d82ca0fdf6ebef5babb813cc11bd8eb2552c846",
        "name": "MAGE English Longformer",
        "ai_label_index": 0, "language": "en",
        "max_tokens": 4096, "inference_batch_size": 1,
        "label_source": "https://github.com/yafuly/MAGE/blob/6d11f851184b9f04166f952ddc1f47727f36710f/README.md",
    },
    "radar-vicuna-en": {
        "repo_id": "TrustSafeAI/RADAR-Vicuna-7B",
        "revision": "4ff1f23a69a36aa1df47b0933be6279f1b896c9b",
        "name": "RADAR Vicuna-trained RoBERTa Large (non-commercial)",
        "ai_label_index": 0, "language": "en",
        "label_source": "https://huggingface.co/spaces/TrustSafeAI/RADAR-AI-Text-Detector/blob/0b17bd20035a44d5bd4df265c03d088ea9389960/app.py",
    },
    "followsci-bert-en": {
        "repo_id": "followsci/bert-ai-text-detector",
        "revision": "dc41bbaff401c56d325f8466d9f8544287669aa1",
        "name": "FollowSci academic paragraph BERT",
        "ai_label_index": 1, "language": "en", "license": "mit",
        "label_source": "https://huggingface.co/followsci/bert-ai-text-detector/blob/dc41bbaff401c56d325f8466d9f8544287669aa1/README.md",
    },
    "roberta-academic-en": {
        "repo_id": "andreas122001/roberta-academic-detector",
        "revision": "f5aa02611897ad2cf981cad63d1f9fd462081414",
        "name": "NTNU academic abstract RoBERTa",
        "ai_label_index": 1, "language": "en", "license": "openrail (model card)",
        "label_source": "https://huggingface.co/andreas122001/roberta-academic-detector/blob/f5aa02611897ad2cf981cad63d1f9fd462081414/config.json",
    },
    "llm-detectaive-en": {
        "repo_id": "raj-tomar001/LLM-DetectAIve", "repo_type": "space",
        "revision": "75c5a84d0815092a455cf9b2b105c0fb4d6277d3",
        "subfolder": "deberta-base",
        "name": "LLM-DetectAIve four-class DeBERTa",
        "num_labels": 4, "ai_label_indices": [1, 2, 3], "language": "en",
        "class_labels": ["human", "human_machine_polished", "machine", "machine_machine_humanized"],
        "score_semantics": "sum_of_nonhuman_classes_including_machine_polishing",
        "score_warning": "LLM-DetectAIve includes machine-polished human writing in ai_score; class_scores retain the distinction. Not a trained insertion-span locator.",
        "license": "not specified in the published Space snapshot",
        "label_source": "https://huggingface.co/spaces/raj-tomar001/LLM-DetectAIve/blob/75c5a84d0815092a455cf9b2b105c0fb4d6277d3/app.py",
    },
    "cocodet-en": {
        "repo_id": "khaaaaaan/CoCoDet",
        "revision": "7f60a5dcdad3af02767cb57da842ead41424c89d",
        "name": "CoCoDet academic peer-review Human/Mix/AI ModernBERT",
        "loader": "cocodet", "weight_file": "CoCoDet.bin",
        "tokenizer_repo_id": "answerdotai/ModernBERT-base",
        "tokenizer_revision": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "config_from_tokenizer_repo": True,
        "num_labels": 3, "ai_label_indices": [1, 2], "language": "en", "license": "mit (model card)",
        "class_labels": ["human", "mix", "ai"],
        "max_tokens": 2048, "inference_batch_size": 1,
        "score_semantics": "sum_of_mix_and_ai_content_classes",
        "score_warning": "CoCoDet targets substantive content in computer-science peer reviews; its human class can include AI language polishing. Transfer to paper bodies is unvalidated.",
        "label_source": "https://github.com/Y1hanChen/COCONUTS/blob/25df60884f344245ce9ddd092055a781919c850b/script/infer.py",
    },
}

# Default inference stays on the existing Chinese roster. A multilingual model
# belongs to both groups and reuses the same local snapshot/cache.
CHINESE_MODELS = tuple(key for key, spec in CATALOG.items() if "zh" in spec["language"].split(","))
ENGLISH_MODELS = ("hc3-en", "desklib-en", "fakespot-en", "bibbbu-xlm-roberta")
MODEL_GROUPS = {"zh": CHINESE_MODELS, "en": ENGLISH_MODELS}

# Additional classifiers are explicit opt-ins until a separately fitted profile
# is available. The existing English four-model profile stays byte-for-byte pinned.
EXTRA_ENGLISH_MODELS = ("openai-base-en", "openai-large-en", "piratexx-en", "mage-en", "radar-vicuna-en")

# Domain/coauthoring checkpoints remain explicit experimental raw-score inputs.
SPECIALIZED_ENGLISH_MODELS = ("followsci-bert-en", "roberta-academic-en", "llm-detectaive-en", "cocodet-en")

# Causal LMs are resources for statistical detectors, not classification scores.
# They are installed with the same manifest but never silently enter fusion.
CAUSAL_ASSETS = {
    "gpt-neo-2.7b": {
        "repo_id": "EleutherAI/gpt-neo-2.7B",
        "revision": "e24fa291132763e59f4a5422741b424fb5d59056",
        "name": "GPT-Neo 2.7B", "kind": "causal_lm", "language": "en",
    },
    "falcon-7b": {
        "repo_id": "tiiuae/falcon-7b",
        "revision": "ec89142b67d748a1865ea4451372db8313ada0d8",
        "name": "Falcon 7B", "kind": "causal_lm", "language": "en",
    },
    "falcon-7b-instruct": {
        "repo_id": "tiiuae/falcon-7b-instruct",
        "revision": "8782b5c5d8c9290412416618f36a133653e85285",
        "name": "Falcon 7B Instruct", "kind": "causal_lm", "language": "en",
    },
}
ASSET_CATALOG = {**CATALOG, **CAUSAL_ASSETS}
STATISTICAL_BACKENDS = {
    "fast-detect-gpt-neo": ("gpt-neo-2.7b",),
    "fast-detect-gpt-falcon": ("falcon-7b", "falcon-7b-instruct"),
    "binoculars-falcon": ("falcon-7b", "falcon-7b-instruct"),
}
