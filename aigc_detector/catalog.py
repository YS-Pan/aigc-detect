"""Explicit label semantics, verified against the authors' inference code/config."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("AIGC_MODEL_DIR", ROOT / "models")).resolve()

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
}
