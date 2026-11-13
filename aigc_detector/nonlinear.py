"""Bounded, score-only nonlinear fusion, evaluated with the standard library."""
import math


def finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def vector(value, size):
    return isinstance(value, list) and len(value) == size and all(finite(x) for x in value)


def validate_head(head, size):
    if not isinstance(head, dict) or not finite(head.get("intercept")):
        raise ValueError("Invalid nonlinear fusion intercept")
    if head.get("kind") == "rbf":
        centers = head.get("centers")
        if (not vector(head.get("linear_coefficients"), size) or not isinstance(centers, list)
                or not 1 <= len(centers) <= 64 or not all(vector(c, size) for c in centers)
                or not vector(head.get("basis_coefficients"), len(centers))
                or not finite(head.get("bandwidth")) or head["bandwidth"] <= 0):
            raise ValueError("Invalid RBF fusion parameters")
    elif head.get("kind") == "tanh_mlp":
        weights = head.get("input_weights")
        if (not isinstance(weights, list) or not 1 <= len(weights) <= 64
                or not all(vector(w, size) for w in weights)
                or not vector(head.get("hidden_bias"), len(weights))
                or not vector(head.get("output_weights"), len(weights))):
            raise ValueError("Invalid MLP fusion parameters")
    else:
        raise ValueError("Unknown nonlinear fusion family")


def evaluate_head(z, head):
    """Return the logit, exact per-input allocation and basis/unit evidence.

    MLP allocations are analytic integrated gradients on the straight path
    from z=0. RBF allocations divide each kernel change by its log-kernel
    coordinate changes. Both add to f(z)-f(0); neither is causal importance.
    """
    allocations = [0.0] * len(z)
    components = []
    if head["kind"] == "rbf":
        for i, (value, coefficient) in enumerate(zip(z, head["linear_coefficients"])):
            contribution = coefficient * value
            allocations[i] += contribution
            components.append({"name": f"linear-{i}", "indices": [i], "value": value,
                               "coefficient": coefficient, "contribution": contribution})
        variance = head["bandwidth"] ** 2
        for j, (center, coefficient) in enumerate(zip(head["centers"], head["basis_coefficients"])):
            origin = math.exp(-math.fsum(c*c for c in center)/(2*variance))
            kernel = math.exp(-math.fsum((v-c)**2 for v, c in zip(z, center))/(2*variance))
            value = kernel-origin
            changes = [(2*v*c-v*v)/(2*variance) for v, c in zip(z, center)]
            total = math.fsum(changes)
            ratio = value/total if abs(total) > 1e-12 else origin
            for i, delta in enumerate(changes):
                allocations[i] += coefficient * ratio * delta
            components.append({"name": f"rbf-{j}", "indices": list(range(len(z))), "value": value,
                               "coefficient": coefficient, "contribution": coefficient*value})
    else:
        for j, (weights, bias, coefficient) in enumerate(zip(head["input_weights"], head["hidden_bias"], head["output_weights"])):
            delta = math.fsum(w*v for w, v in zip(weights, z))
            origin = math.tanh(bias)
            value = math.tanh(bias+delta)-origin
            ratio = value/delta if abs(delta) > 1e-12 else 1-origin*origin
            for i, (weight, v) in enumerate(zip(weights, z)):
                allocations[i] += coefficient * ratio * weight * v
            components.append({"name": f"hidden-{j}", "indices": list(range(len(z))), "value": value,
                               "coefficient": coefficient, "contribution": coefficient*value})
    logit = head["intercept"] + math.fsum(c["contribution"] for c in components)
    if not math.isfinite(logit) or not all(math.isfinite(v) for v in allocations):
        raise ValueError("Non-finite nonlinear fusion output")
    # Preserve floating-point allocation accounting without changing the score.
    residual = logit-head["intercept"]-math.fsum(allocations)
    if abs(residual) > 1e-8 * max(1, abs(logit), math.fsum(abs(v) for v in allocations)):
        raise ValueError("Nonlinear allocation accounting failed")
    allocations[max(range(len(z)), key=lambda i: abs(allocations[i]))] += residual
    return logit, allocations, components
