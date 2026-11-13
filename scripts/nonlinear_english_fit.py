"""CPU fitting of RBF kernels and small jointly optimized neural fusers."""
import math
import numpy as np

from scripts.calibrate_english_v2 import transformed, CLASSIFIERS
from scripts.fit_fusion_v2 import fit_head, sigmoid


def standardize(records, signals, human_weight):
    totals = {}
    for r in records:
        key = r["paper_id"], r["ai_chars"] == 0
        totals[key] = totals.get(key, 0) + r["chars"]
    weights = np.array([r["chars"]/totals[r["paper_id"], r["ai_chars"] == 0] *
                        (human_weight if r["ai_chars"] == 0 else 1.) for r in records])
    raw = transformed(records, signals)
    center = np.average(raw, axis=0, weights=weights)
    floor = np.array([.01 if s == "binoculars-falcon" else .5 for s in signals])
    scale = np.maximum(np.sqrt(np.average((raw-center)**2, axis=0, weights=weights)), floor)
    return center, scale, weights/weights.mean(), np.clip((raw-center)/scale, -5., 5.)


def transform(records, model):
    return np.clip((transformed(records, model["signals"])-model["center"])/model["scale"], -5., 5.)


def landmarks(z, count):
    """Deterministic farthest initialization and 12 Lloyd updates, fit scores only."""
    count = min(count, len(z))
    centers = [z[np.argmin(np.sum(z*z, axis=1))].copy()]
    distance = np.full(len(z), np.inf)
    for _ in range(1, count):
        distance = np.minimum(distance, np.sum((z-centers[-1])**2, axis=1))
        centers.append(z[np.argmax(distance)].copy())
    centers = np.asarray(centers)
    for _ in range(12):
        labels = np.argmin(np.sum((z[:, None, :]-centers[None, :, :])**2, axis=2), axis=1)
        updated = np.array([z[labels == i].mean(axis=0) if np.any(labels == i) else c for i, c in enumerate(centers)])
        if np.max(np.abs(updated-centers)) < 1e-8:
            break
        centers = updated
    return centers


def rbf_features(z, centers, bandwidth):
    kernel = np.exp(-np.sum((z[:, None, :]-centers[None, :, :])**2, axis=2)/(2*bandwidth**2))
    origin = np.exp(-np.sum(centers*centers, axis=1)/(2*bandwidth**2))
    return np.column_stack([np.ones(len(z)), z, kernel-origin])


def predict_z(z, head):
    if head["kind"] == "rbf":
        parameters = np.array([head["intercept"]] + head["linear_coefficients"] + head["basis_coefficients"])
        return rbf_features(z, np.array(head["centers"]), head["bandwidth"]) @ parameters
    bias = np.array(head["hidden_bias"])
    hidden = np.tanh(z @ np.array(head["input_weights"]).T+bias)-np.tanh(bias)
    return head["intercept"] + hidden @ np.array(head["output_weights"])


def predict(records, model):
    return predict_z(transform(records, model), model["head"])


def targets(records, mode):
    return np.array([r["ai_chars"]/r["chars"] if mode == "fraction" else float(r["ai_chars"] > 0) for r in records])


def fit_rbf(z, y, weights, config, centers):
    bandwidth = config["width"] * math.sqrt(z.shape[1])
    x = rbf_features(z, centers, bandwidth)
    parameters, info = fit_head(x, y, weights, [[i] for i in range(x.shape[1]-1)], config["regularization"])
    head = {"kind": "rbf", "intercept": float(parameters[0]), "linear_coefficients": parameters[1:1+z.shape[1]].tolist(),
            "centers": centers.tolist(), "bandwidth": bandwidth, "basis_coefficients": parameters[1+z.shape[1]:].tolist()}
    return head, {**info, "landmarks": "Deterministic farthest initialization + fit-only Lloyd updates",
                  "parameter_count": len(parameters)}


def fit_mlp(z, y, weights, config, optimizer):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    x, target, w = [torch.as_tensor(v, dtype=torch.float64) for v in (z, y, weights)]
    ai, human = target > 0, target == 0
    all_starts = []
    for seed in optimizer["seeds"]:
        generator = torch.Generator().manual_seed(seed)
        hidden = config["hidden"]
        input_weights = torch.nn.Parameter(torch.randn(hidden, z.shape[1], generator=generator, dtype=torch.float64)/math.sqrt(z.shape[1]))
        hidden_bias = torch.nn.Parameter(torch.zeros(hidden, dtype=torch.float64))
        output_weights = torch.nn.Parameter(torch.randn(hidden, generator=generator, dtype=torch.float64)*.05)
        mean = min(1-1e-5, max(1e-5, float(np.average(y, weights=weights))))
        intercept = torch.nn.Parameter(torch.tensor(math.log(mean/(1-mean)), dtype=torch.float64))
        parameters = [input_weights, hidden_bias, output_weights, intercept]
        def loss():
            values = intercept + (torch.tanh(x @ input_weights.T+hidden_bias)-torch.tanh(hidden_bias)) @ output_weights
            data = torch.sum(w * F.binary_cross_entropy_with_logits(values, target, reduction="none"))
            penalty = .5 * config["regularization"] * (input_weights.square().sum()+output_weights.square().sum()+.1*hidden_bias.square().sum())
            result = (data+penalty+.5e-6*intercept.square())/len(z)
            if config["target_mode"] == "ranking_presence":
                pairs = F.softplus(optimizer["ranking_margin"]+values[human][None, :]-values[ai][:, None])
                pair_weights = w[ai][:, None]*w[human][None, :]
                result = result + optimizer["ranking_weight"] * (pairs*pair_weights).sum()/pair_weights.sum()
            return result
        initial = float(loss().detach())
        adam = torch.optim.Adam(parameters, lr=optimizer["adam_learning_rate"])
        trace = []
        for step in range(optimizer["adam_steps"]):
            adam.zero_grad()
            value = loss()
            value.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.)
            adam.step()
            if step % 50 == 0:
                trace.append({"adam_step": step, "objective": float(value.detach())})
        lbfgs = torch.optim.LBFGS(parameters, max_iter=optimizer["lbfgs_steps"], tolerance_grad=1e-7,
                                 tolerance_change=1e-11, history_size=20, line_search_fn="strong_wolfe")
        evaluations = 0
        def closure():
            nonlocal evaluations
            lbfgs.zero_grad()
            value = loss()
            value.backward()
            evaluations += 1
            return value
        lbfgs.step(closure)
        value = closure()
        final = float(value.detach())
        gradient = max(float(p.grad.detach().abs().max()) for p in parameters)
        if not math.isfinite(final) or not math.isfinite(gradient) or final > initial+1e-8:
            raise ValueError("Non-finite or worsening neural optimization")
        head = {"kind": "tanh_mlp", "intercept": float(intercept.detach()),
                "input_weights": input_weights.detach().numpy().tolist(), "hidden_bias": hidden_bias.detach().numpy().tolist(),
                "output_weights": output_weights.detach().numpy().tolist()}
        all_starts.append({"head": head, "seed": seed, "initial_objective": initial, "final_objective": final,
                           "final_gradient_inf_norm": gradient, "lbfgs_closure_evaluations": evaluations, "trace": trace})
    chosen = min(all_starts, key=lambda s: (s["final_objective"], s["seed"]))
    info = {"method": "Adam initialization + L-BFGS with strong-Wolfe line search; float64 CPU",
            "parameter_count": 1+config["hidden"]*(z.shape[1]+2), "chosen_seed": chosen["seed"],
            "final_gradient_inf_norm": chosen["final_gradient_inf_norm"],
            "stationary_to_1e-4": chosen["final_gradient_inf_norm"] < 1e-4,
            "global_optimum_claimed": False, "starts": [{k: v for k, v in s.items() if k != "head"} for s in all_starts]}
    return chosen["head"], info


def fit(records, signals, config, optimizer, prepared=None):
    center, scale, weights, z = prepared or standardize(records, signals, config["human_weight"])
    y = targets(records, config["target_mode"])
    if config["family"] == "rbf":
        head, info = fit_rbf(z, y, weights, config, landmarks(z, config["centers"]))
    else:
        head, info = fit_mlp(z, y, weights, config, optimizer)
    return {"signals": list(signals), "center": center.tolist(), "scale": scale.tolist(), "head": head, "optimization": info}
