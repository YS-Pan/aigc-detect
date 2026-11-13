import sys

import pytest

from aigc_detector import cli, network, prepare
from aigc_detector.catalog import EXTRA_ENGLISH_MODELS, MODEL_GROUPS, SPECIALIZED_ENGLISH_MODELS, STATISTICAL_BACKENDS
from scripts import download_models, network as legacy_network


@pytest.mark.parametrize("arguments,expected", [
    ([], list(MODEL_GROUPS["zh"])),
    (["--language", "en"], list(MODEL_GROUPS["en"])),
    (["--extra-english"], list(EXTRA_ENGLISH_MODELS)),
    (["--specialized-english"], list(SPECIALIZED_ENGLISH_MODELS)),
    (["--extra-english", "--specialized-english"], list(SPECIALIZED_ENGLISH_MODELS)),
    (["--backends", "binoculars-falcon"], list(STATISTICAL_BACKENDS["binoculars-falcon"])),
    (["--models", "hc3-en", "--backends", "binoculars-falcon", "fast-detect-gpt-falcon"],
     ["hc3-en", *STATISTICAL_BACKENDS["binoculars-falcon"]]),
])
def test_legacy_options_delegate_one_complete_selection(monkeypatch, arguments, expected):
    captured = []
    original = ["download_models.py", *arguments, "--download-workers", "3"]
    monkeypatch.setattr(sys, "argv", original)
    monkeypatch.setattr(cli, "main", lambda: captured.append(list(sys.argv)) or 0)
    assert download_models.main() == 0
    assert captured == [["aigc-detect", "prepare", "--models", *expected, "--download-workers", "3"]]
    assert sys.argv is original


def test_no_config_noninteractive_does_not_download(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AIGC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("AIGC_MODEL_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--extra-english"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(prepare, "get_json", lambda *a, **k: pytest.fail("unexpected network call"))
    assert download_models.main() == 1
    assert "PreparationRequired" in capsys.readouterr().err
    assert not (tmp_path / "state").exists()


def test_selected_directory_passes_through_same_preparation(tmp_path, monkeypatch):
    model_dir = tmp_path / "selected models"
    monkeypatch.setenv("AIGC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("AIGC_MODEL_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--models", "hc3-en", "--assets-dir", str(model_dir)])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    calls = []
    monkeypatch.setattr(prepare, "prepare", lambda root, models, **kwargs: calls.append((root, models, kwargs)))
    assert download_models.main() == 0
    assert calls == [(model_dir, ["hc3-en"], {"download_workers": 1})]


def test_legacy_imports_are_the_package_implementation():
    assert download_models.model_assets is prepare.model_assets
    assert legacy_network.download is network.download
    assert legacy_network.open_public is network.open_public
    assert legacy_network.EVENTS is network.EVENTS
