"""SNR metrics and statistical tests.

SNR_act = mean(activation_diff) / std(activation_diff)
SNR_J = mean(jacobian_diff) / std(jacobian_diff)
ratio = SNR_J / SNR_act

GO decision:     ratio > 3
MARGINAL:         1 < ratio <= 3
NO-GO:            ratio <= 1
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def compute_activation_diff(normal_activations, perturbed_activations):
    """Compute per-prompt RELATIVE activation differences at the last token.

    Formula from research plan:
        act_diff_i = ||h_anom_i - h_norm_i|| / ||h_norm_i||

    This is dimensionless (fractional change) and comparable across layers/models.

    Args:
        normal_activations: dict[layer_idx -> list of tensors (hidden_dim,)]
        perturbed_activations: same structure

    Returns:
        dict[layer_idx -> np.array of relative L2 norm differences]
    """
    diffs = {}
    for layer_idx in normal_activations:
        n_list = normal_activations[layer_idx]
        p_list = perturbed_activations[layer_idx]
        layer_diffs = []
        for n_act, p_act in zip(n_list, p_list):
            norm_clean = n_act.float().norm().item()
            diff_abs = (p_act - n_act).float().norm().item()
            # Relative difference: ||Δh|| / ||h||
            diff = diff_abs / (norm_clean + 1e-10)
            layer_diffs.append(diff)
        diffs[layer_idx] = np.array(layer_diffs)
    return diffs


def compute_jvp_diff(normal_jvps, perturbed_jvps):
    """Compute per-prompt RELATIVE JVP norm differences.

    Formula from research plan:
        jvp_diff_i = mean_k(|jvp_anom_ik - jvp_norm_ik|) / mean_k(jvp_norm_ik)

    This is dimensionless (fractional change in JVP norms).

    Args:
        normal_jvps: dict[layer_idx -> list of tensors (k,)]
        perturbed_jvps: same structure

    Returns:
        dict[layer_idx -> np.array of relative mean JVP differences per prompt]
    """
    diffs = {}
    for layer_idx in normal_jvps:
        n_list = normal_jvps[layer_idx]
        p_list = perturbed_jvps[layer_idx]
        layer_diffs = []
        for n_jvp, p_jvp in zip(n_list, p_list):
            n_arr = n_jvp.numpy()
            p_arr = p_jvp.numpy()
            # Mean absolute difference across k directions, normalized by mean clean JVP
            abs_diff = np.abs(p_arr - n_arr).mean()
            mean_clean = np.mean(n_arr)
            diff = abs_diff / (mean_clean + 1e-10)
            layer_diffs.append(diff)
        diffs[layer_idx] = np.array(layer_diffs)
    return diffs


def compute_jacobian_frobenius_diff(normal_jacobians, perturbed_jacobians):
    """Compute per-prompt relative Jacobian Frobenius norm difference.

    Formula: ||J_anom - J_norm||_F / ||J_norm||_F

    Uses the FULL d×d Jacobian matrix — no random projections, no scalar summaries.
    This matches the research plan's formula exactly.

    Args:
        normal_jacobians: dict[layer_idx -> list of (d, d) tensors]
        perturbed_jacobians: same structure

    Returns:
        dict[layer_idx -> np.array of relative Frobenius differences per prompt]
    """
    import torch
    diffs = {}
    for layer_idx in normal_jacobians:
        n_list = normal_jacobians[layer_idx]
        p_list = perturbed_jacobians[layer_idx]
        layer_diffs = []
        for n_jac, p_jac in zip(n_list, p_list):
            diff_frob = torch.norm(p_jac - n_jac, p='fro').item()
            clean_frob = torch.norm(n_jac, p='fro').item()
            layer_diffs.append(diff_frob / (clean_frob + 1e-10))
        diffs[layer_idx] = np.array(layer_diffs)
    return diffs


def snr(values):
    """Signal-to-noise ratio: mean / std.

    Returns 0 if std is 0 (no variation).
    """
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0 if mean == 0 else float('inf')
    return mean / std


def compute_snr_metrics(act_diffs, jvp_diffs):
    """Compute SNR metrics for all layers.

    Args:
        act_diffs: dict[layer_idx -> np.array of activation diffs]
        jvp_diffs: dict[layer_idx -> np.array of JVP diffs]

    Returns:
        dict[layer_idx -> dict with keys:
            'snr_act', 'snr_j', 'ratio', 'decision',
            'act_mean', 'act_std', 'jvp_mean', 'jvp_std',
            'welch_t', 'welch_p']
    """
    results = {}
    for layer_idx in act_diffs:
        act = act_diffs[layer_idx]
        jvp_d = jvp_diffs[layer_idx]

        snr_act = snr(act)
        snr_j = snr(jvp_d)

        if snr_act == 0:
            ratio = float('inf') if snr_j > 0 else 0.0
        else:
            ratio = snr_j / snr_act

        # Decision
        if ratio > 3:
            decision = "GO"
        elif ratio > 1:
            decision = "MARGINAL"
        else:
            decision = "NO-GO"

        # Welch's t-test: are JVP diffs significantly different from activation diffs?
        # Normalize both to [0,1] range for fair comparison
        if len(act) > 1 and len(jvp_d) > 1:
            act_norm = act / (act.max() + 1e-10)
            jvp_norm = jvp_d / (jvp_d.max() + 1e-10)
            t_stat, p_val = stats.ttest_ind(jvp_norm, act_norm, equal_var=False)
        else:
            t_stat, p_val = 0.0, 1.0

        # ROC AUC: can we distinguish "clean" from "perturbed" using act or JVP?
        # Create synthetic clean samples: mirror around zero (diff = 0 for clean)
        n = len(act)
        labels = np.concatenate([np.zeros(n), np.ones(n)])  # 0=clean, 1=perturbed
        # For activations: use act_diff as score (higher = more likely perturbed)
        act_scores = np.concatenate([np.zeros(n), act])
        # For JVP: use jvp_diff as score
        jvp_scores = np.concatenate([np.zeros(n), jvp_d])
        try:
            auc_act = roc_auc_score(labels, act_scores)
        except ValueError:
            auc_act = 0.5
        try:
            auc_jvp = roc_auc_score(labels, jvp_scores)
        except ValueError:
            auc_jvp = 0.5

        results[layer_idx] = {
            "snr_act": snr_act,
            "snr_j": snr_j,
            "ratio": ratio,
            "decision": decision,
            "act_mean": float(np.mean(act)),
            "act_std": float(np.std(act)),
            "jvp_mean": float(np.mean(jvp_d)),
            "jvp_std": float(np.std(jvp_d)),
            "welch_t": float(t_stat),
            "welch_p": float(p_val),
            "auc_act": float(auc_act),
            "auc_jvp": float(auc_jvp),
        }

    return results


def aggregate_decision(all_results):
    """Aggregate per-condition results into an overall GO/NO-GO decision.

    Logic: if ANY condition shows GO for majority of layers -> overall GO.
    """
    go_count = 0
    marginal_count = 0
    nogo_count = 0

    for condition_key, layer_results in all_results.items():
        for layer_idx, metrics in layer_results.items():
            d = metrics["decision"]
            if d == "GO":
                go_count += 1
            elif d == "MARGINAL":
                marginal_count += 1
            else:
                nogo_count += 1

    total = go_count + marginal_count + nogo_count
    if total == 0:
        return "NO-GO", {}

    summary = {
        "go_count": go_count,
        "marginal_count": marginal_count,
        "nogo_count": nogo_count,
        "go_fraction": go_count / total,
    }

    if go_count / total > 0.5:
        return "GO", summary
    elif (go_count + marginal_count) / total > 0.5:
        return "MARGINAL", summary
    else:
        return "NO-GO", summary


def format_results_table(results, condition_key):
    """Format results as a readable table string."""
    lines = [f"\n{'='*70}"]
    lines.append(f"Condition: {condition_key}")
    lines.append(f"{'='*70}")
    lines.append(f"{'Layer':>6} | {'SNR_act':>10} | {'SNR_J':>10} | {'Ratio':>10} | {'Decision':>10} | {'Welch p':>10}")
    lines.append(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for layer_idx in sorted(results.keys()):
        m = results[layer_idx]
        lines.append(
            f"{layer_idx:>6} | {m['snr_act']:>10.3f} | {m['snr_j']:>10.3f} | "
            f"{m['ratio']:>10.3f} | {m['decision']:>10} | {m['welch_p']:>10.4f}"
        )

    return "\n".join(lines)
