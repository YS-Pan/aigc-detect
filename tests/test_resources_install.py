import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aigc_detector import cli
from aigc_detector.paths import state_dir
from aigc_detector.resources import missing, selected_assets, verify


def fake_asset(root, model="mpu-zh-v3"):
    directory = root / model
    directory.mkdir(parents=True)
    entries = {}
    for name, content in (("config.json", b"{}"), ("pytorch_model.bin", b"test")):
        (directory / name).write_bytes(content)
        entries[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (root / "manifest.json").write_text(json.dumps({"models": {model: {"installed": True, "revision": "test", "files": entries}}}), encoding="utf-8")
    return model


def test_inventory_and_verify(tmp_path):
    model = fake_asset(tmp_path)
    assert not missing(tmp_path, [model])
    verify(tmp_path, [model])
    (tmp_path / model / "pytorch_model.bin").write_bytes(b"fail")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(tmp_path, [model])


def test_explicit_state_does_not_require_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGC_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: pytest.fail("explicit state must not resolve home"))
    assert state_dir() == tmp_path.resolve()


def test_feature_rosters_are_selected_without_torch():
    assert len(selected_assets(annotation="zh")) == 10
    assert len(selected_assets(annotation="en")) == 5
    assert selected_assets(backends=["binoculars-falcon"]) == ["falcon-7b", "falcon-7b-instruct"]


def test_noninteractive_missing_fails_without_network(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AIGC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("AIGC_MODEL_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["aigc-detect", "detect", "--text", "test", "--assets-dir", str(tmp_path / "absent")])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli.main() == 1
    assert '"error": "PreparationRequired"' in capsys.readouterr().err
    assert not (tmp_path / "state").exists()
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("command", [
    ["detect", "--text", "test", "--models", "falcon-7b"],
    ["annotate", "--text", "test", "--models", "falcon-7b", "--report-dir", "report"],
    ["annotate", "--text", "test", "--language", "en", "--models", "desklib-en", "--report-dir", "report"],
])
def test_invalid_classifier_or_profile_fails_before_preparation(tmp_path, monkeypatch, capsys, command):
    monkeypatch.setenv("AIGC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("AIGC_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["aigc-detect", *command])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("invalid request must not prompt for downloads"))
    assert cli.main() == 1
    assert '"error": "ValueError"' in capsys.readouterr().err
    assert not (tmp_path / "state").exists()


def test_interactive_cancel_makes_no_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AIGC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("AIGC_MODEL_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["aigc-detect", "prepare", "--models", "mpu-zh-v3"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "q")
    assert cli.main() == 1
    assert not (tmp_path / "state").exists()


def test_stdin_payload_never_prompts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AIGC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("AIGC_MODEL_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["aigc-detect", "detect", "--stdin"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("stdin belongs to the text payload"))
    assert cli.main() == 1
    assert '"error": "PreparationRequired"' in capsys.readouterr().err
    assert not (tmp_path / "state").exists()


def test_explicit_prepare_uses_exact_model_directory(tmp_path, monkeypatch):
    from aigc_detector import prepare
    root, state = tmp_path / "chosen models", tmp_path / "state"
    calls = []
    monkeypatch.setenv("AIGC_STATE_DIR", str(state))
    monkeypatch.setattr(sys, "argv", ["aigc-detect", "prepare", "--models", "mpu-zh-v3", "--assets-dir", str(root)])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(prepare, "prepare", lambda directory, selected, **kwargs: calls.append((directory, selected)))
    assert cli.main() == 0
    assert calls == [(root, ["mpu-zh-v3"])]
    assert json.loads((state / "config.json").read_text(encoding="utf-8"))["model_dir"] == str(root)


def test_reuse_complete_model_without_network(tmp_path, monkeypatch):
    from aigc_detector import prepare
    model = fake_asset(tmp_path)
    monkeypatch.setattr(prepare, "get_json", lambda *a, **k: pytest.fail("unexpected network call"))
    prepare.prepare(tmp_path, [model])


def test_light_commands_without_site_packages(tmp_path):
    # -S suppresses third-party site-packages, including Torch.
    package_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(package_root), "AIGC_STATE_DIR": str(tmp_path / "state")}
    for command in (["--help"], ["models"], ["prepare", "--help"], ["verify", "--help"]):
        result = subprocess.run([sys.executable, "-S", "-m", "aigc_detector", *command], cwd=tmp_path,
                                env=env, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr
    assert not (tmp_path / "state").exists()


def test_light_inference_reports_missing_extra(tmp_path):
    root = tmp_path / "models"
    model = fake_asset(root)
    package_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(package_root), "AIGC_STATE_DIR": str(tmp_path / "state")}
    result = subprocess.run([sys.executable, "-S", "-m", "aigc_detector", "detect", "--text", "test",
                             "--models", model, "--assets-dir", str(root)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 1
    assert "local-aigc-detectors[inference]" in result.stderr


def test_help_and_explicit_model_dir_do_not_resolve_home(tmp_path):
    package_root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(package_root)}
    env.pop("AIGC_STATE_DIR", None)
    env.pop("AIGC_MODEL_DIR", None)
    code = ("import runpy,sys; from pathlib import Path; "
            "Path.home=classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError('home unavailable'))); "
            "sys.argv=['aigc-detect',*sys.argv[1:]]; runpy.run_module('aigc_detector',run_name='__main__')")
    for command in (["--help"], ["models", "--assets-dir", str(tmp_path / "models")]):
        result = subprocess.run([sys.executable, "-S", "-c", code, *command], cwd=tmp_path,
                                env=env, capture_output=True, text=True, encoding="utf-8")
        assert result.returncode == 0, result.stderr
