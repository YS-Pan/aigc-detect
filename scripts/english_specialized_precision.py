"""Second declared diagnostic: retain small specialized probabilities.

Uses the first experiment's exact split/selection loop under scoped adapters.
Its frozen files remain unchanged. This follow-up is informed by the first
experiment and its repeat results; neither is independent validation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import english_specialized_fusion as sf
from scripts import english_hard_negative as hn

PARENT = sf.EVIDENCE
EVIDENCE = ROOT / "evaluations/english-specialized-precision-20261004"
REPORTS = ROOT / "reports/english-specialized-precision-20261004"
BASE_CONFIGURATIONS = sf.configurations
BASE_PREDICT = hn.predict


def configurations():
    result = []
    for config in BASE_CONFIGURATIONS():
        transforms = ("native_logit",) if config["method"] == "single" else ("native_logit", "raw_probability")
        for transform in transforms:
            result.append({**config, "id": transform + "-" + config["id"],
                           "family": transform + "-" + config["family"], "transform": transform})
    return result


def raw_matrix(rows, signals, transform):
    p = np.asarray([[r["scores"][s] for s in signals] for r in rows], dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid raw scores")
    if transform == "raw_probability":
        return p
    if transform == "native_logit":
        # Preserve every representable small output; exact 0/1 remain finite.
        return np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(1-p, 1e-30))
    raise ValueError(transform)


def predict(rows, model):
    if "transform" not in model:
        return BASE_PREDICT(rows, model)
    z = np.clip((raw_matrix(rows, model["signals"], model["transform"]) - model["center"]) / model["scale"],
                -model["clip_standardized"], model["clip_standardized"])
    return z @ np.asarray(model["coefficients"]) + model["intercept"]


def fit_model(data, papers, config, campaign):
    transform, signals = config["transform"], config["signals"]
    if config["method"] == "single":
        assert transform == "native_logit"
        return {"signals": signals, "transform": transform, "clip_standardized": 70.,
                "center": [0.], "scale": [1.], "coefficients": [1.], "intercept": 0.,
                "configuration": config, "fit_papers": sorted(papers),
                "optimization": {"stationary_1e_5": True, "gradient_inf": 0.}}
    rows, bags, bag_papers, excluded = hn.training_view(data, set(papers))
    raw = raw_matrix(rows, signals, transform)
    weights = hn.balanced_weights(rows)
    center = np.average(raw, axis=0, weights=weights)
    floor = .5 if transform == "native_logit" else 1e-12
    scale = np.maximum(np.sqrt(np.average((raw-center)**2, axis=0, weights=weights)), floor)
    z = np.clip((raw-center)/scale, -5, 5)
    n = z.shape[1]
    theta0, info0 = hn.optimize(hn.make_loss(z, rows, bags, bag_papers, {**config, "method": "presence"}, campaign), np.zeros(n+1), campaign)
    if config["method"] == "presence":
        theta, info, starts = theta0, info0, [info0]
    else:
        loss = hn.make_loss(z, rows, bags, bag_papers, config, campaign)
        candidates = [hn.optimize(loss, start, campaign) for start in (np.zeros(n), theta0[:n])]
        theta, info = min(candidates, key=lambda p: p[1]["final_loss"])
        starts = [item[1] for item in candidates]
    return {"signals": signals, "transform": transform, "clip_standardized": 5.,
            "center": center.tolist(), "scale": scale.tolist(), "coefficients": theta[:n].tolist(),
            "intercept": float(theta[-1]) if config["method"] == "presence" else 0.,
            "configuration": config, "fit_papers": sorted(papers), "fit_unique_windows": len(rows),
            "fit_bags": len(bags), "excluded_conflicting_window_ids": excluded,
            "optimization": {**info, "starts": starts}}


def prepare():
    sf.prepare()
    campaign = sf.read(EVIDENCE / "campaign.json")
    campaign.update({
        "schema_version": "english-specialized-precision-experiment/1.0",
        "reason": "First experiment's inherited 1e-5 logit clipping collapsed all FollowSci outputs and most CoCoDet class outputs. Preserve small probabilities and compare raw-probability linear models as a bounded follow-up.",
        "transforms": {"native_logit": "log(max(p,1e-30))-log(max(1-p,1e-30)); weighted SD floor .5",
                       "raw_probability": "p without logit; weighted SD floor 1e-12"},
        "standardization": "Fit-only equal paper/class center and scale; transformed SD floors above; standardized clip 5. Single-signal controls retain full native logit (clip 70) and same .01 margin.",
        "new_base_inference": False, "prior_repeat_results_known": True,
        "repeat_rule": "Re-freeze before loading the repeat score file for this follow-up; scores and historical failures already known, not a new independent test.",
    })
    for path in (Path(__file__), PARENT / "development-data.json", PARENT / "repeat-data.json", PARENT / "FREEZE.json", PARENT / "evaluation.json"):
        campaign["source_hashes"][path.relative_to(ROOT).as_posix()] = sf.sha(path)
    sf.write(EVIDENCE / "campaign.json", campaign)
    dev = copy.deepcopy(sf.read(PARENT / "development-data.json"))
    dev["config"]["source_data_sha256"] = sf.sha(PARENT / "development-data.json")
    dev["config"]["campaign_sha256"] = sf.sha(EVIDENCE / "campaign.json")
    sf.write(EVIDENCE / "development-data.json", dev)
    print(json.dumps({"precision_campaign_declared": True, "candidate_count": len(configurations())}), flush=True)


def repeat():
    sf.verify_freeze()
    if (EVIDENCE / "repeat-data.json").exists():
        raise FileExistsError("Preserve precision repeat data")
    data = copy.deepcopy(sf.read(PARENT / "repeat-data.json"))
    data["config"].update(source_data_sha256=sf.sha(PARENT / "repeat-data.json"),
                          campaign_sha256=sf.sha(EVIDENCE / "campaign.json"),
                          freeze_sha256=sf.sha(EVIDENCE / "FREEZE.json"))
    sf.write(EVIDENCE / "repeat-data.json", data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fit", "repeat", "evaluate", "verify"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    with patch.object(sf, "EVIDENCE", EVIDENCE), patch.object(sf, "REPORTS", REPORTS), \
            patch.object(sf, "configurations", configurations), patch.object(sf, "fit_model", fit_model), \
            patch.object(hn, "predict", predict):
        {"prepare": prepare, "fit": sf.fit, "repeat": repeat, "evaluate": sf.evaluate, "verify": sf.verify_result}[args.action]()


if __name__ == "__main__":
    main()
