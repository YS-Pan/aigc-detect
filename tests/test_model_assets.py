from scripts.download_models import model_assets


def test_selects_only_requested_checkpoint_and_prefers_safetensors():
    selected = "chinese-bert-sft"
    names = ["README.md", "other/model.safetensors", "qwen/model.safetensors",
             f"{selected}/config.json", f"{selected}/model.safetensors",
             f"{selected}/pytorch_model.bin", f"{selected}/modeling.py",
             f"{selected}/nested/model.safetensors", f"{selected}/../config.json"]
    assets = model_assets({"siblings": [{"rfilename": name} for name in names]}, selected)
    assert {name for name, _ in assets} == {"config.json", "model.safetensors"}
    assert all(item["rfilename"].startswith(selected + "/") for _, item in assets)
