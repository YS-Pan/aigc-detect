# Smoke test samples

These samples test installation, inference, offline operation and long-text coverage. They do not establish thresholds or detection accuracy.

- `mpu-official-demo-zh`: Chinese text copied from the author's demo, no ground-truth label.
- Six HC3-Chinese answers: first three baike rows with both human and ChatGPT answers of at least 100 characters. Labels preserve the source answer field; they are not judgments made by this wrapper.
- `long-window-coverage`: six answers concatenated and repeated three times to exercise multiple windows.

HC3 source: https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese/resolve/09a687b8dc164b89e7df95abf15df3b216bc31c2/baike.jsonl

MPU demo source: https://huggingface.co/spaces/yuchuantian/AIGC_text_detector/raw/191c98291f204cae3e52db965ab0f944c9005067/app.py

HC3 attribution: Biyang Guo et al., How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection (2023). Dataset license: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/. The derived fixture and HC3 excerpts retain this license. The released HC3 detector was trained on the entire HC3-Chinese dataset without a held-out split. These samples therefore cannot serve as an independent accuracy benchmark. Exact revisions, answer indices and content hashes are in smoke.jsonl.
