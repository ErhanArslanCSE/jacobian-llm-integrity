"""Statistical metrics for NIAC detection.

Replaces exp1's SNR ratio with proper hypothesis testing:
- One-sample t-test: H0: mu=0 vs H1: mu>0
- NIAC Score: J(x) - rho * A(x) (Neyman-Pearson optimal)
- AUC-ROC: Clean vs perturbed discrimination
- F-test: Variance inflation detection
- Temporal accumulation: Cumulative SNR vs observation count
- Permutation test: Non-parametric p-value
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


# ── Activation and Jacobian difference computation ──

def compute_activation_diff(normal_activations, perturbed_activations):
    """Compute per-prompt RELATIVE activation differences at the last token.

    Formula: act_diff_i = ||h_anom_i - h_norm_i|| / ||h_norm_i||

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
            diff = diff_abs / (norm_clean + 1e-10)
            layer_diffs.append(diff)
        diffs[layer_idx] = np.array(layer_diffs)
    return diffs


def compute_jacobian_frobenius_diff(normal_jacobians, perturbed_jacobians):
    """Compute per-prompt relative Jacobian Frobenius norm difference.

    Formula: ||J_anom - J_norm||_F / ||J_norm||_F

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


# ── Core statistical tests ──

def one_sample_ttest(values, alternative="greater"):
    """One-sample t-test: H0: mu=0, H1: mu>0 (one-sided).

    Args:
        values: np.array of observations
        alternative: "greater" (default) or "two-sided"

    Returns:
        dict with t_stat, p_value, mean, std, n, ci_95_lower
    """
    n = len(values)
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "mean": 0.0, "std": 0.0,
                "n": n, "ci_95_lower": 0.0}

    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)

    if se < 1e-15:
        # All values identical
        if mean > 0:
            return {"t_stat": float('inf'), "p_value": 0.0, "mean": float(mean),
                    "std": 0.0, "n": n, "ci_95_lower": float(mean)}
        else:
            return {"t_stat": 0.0, "p_value": 1.0, "mean": float(mean),
                    "std": 0.0, "n": n, "ci_95_lower": float(mean)}

    t_stat = mean / se
    if alternative == "greater":
        p_value = 1.0 - stats.t.cdf(t_stat, df=n - 1)
    else:
        p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1))

    # 95% CI lower bound (one-sided)
    t_crit = stats.t.ppf(0.95, df=n - 1)
    ci_lower = mean - t_crit * se

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean": float(mean),
        "std": float(std),
        "n": n,
        "ci_95_lower": float(ci_lower),
    }


def f_test_variance(values1, values2):
    """F-test for variance comparison.

    Tests whether values1 has significantly higher variance than values2.
    H0: sigma1^2 = sigma2^2, H1: sigma1^2 > sigma2^2

    Args:
        values1: np.array (expected higher variance)
        values2: np.array (expected lower variance)

    Returns:
        dict with f_stat, p_value, var1, var2
    """
    n1, n2 = len(values1), len(values2)
    if n1 < 2 or n2 < 2:
        return {"f_stat": 1.0, "p_value": 1.0, "var1": 0.0, "var2": 0.0}

    var1 = np.var(values1, ddof=1)
    var2 = np.var(values2, ddof=1)

    if var2 < 1e-15:
        f_stat = float('inf') if var1 > 0 else 1.0
        p_value = 0.0 if var1 > 0 else 1.0
    else:
        f_stat = var1 / var2
        p_value = 1.0 - stats.f.cdf(f_stat, dfn=n1 - 1, dfd=n2 - 1)

    return {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "var1": float(var1),
        "var2": float(var2),
    }


def compute_auc(clean_scores, perturbed_scores):
    """Compute AUC-ROC for binary discrimination.

    Clean samples get label 0, perturbed get label 1.
    Higher score = more likely perturbed.

    Args:
        clean_scores: np.array of scores under clean model
        perturbed_scores: np.array of scores under perturbed model

    Returns:
        float: AUC-ROC score
    """
    n_clean = len(clean_scores)
    n_pert = len(perturbed_scores)
    labels = np.concatenate([np.zeros(n_clean), np.ones(n_pert)])
    scores = np.concatenate([clean_scores, perturbed_scores])
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def niac_score(jac_diffs, act_diffs, rho=1.0):
    """NIAC Score: Neyman-Pearson optimal test statistic.

    NIAC(x) = J(x) - rho * A(x)

    When rho is chosen so that activation changes are penalized,
    the NIAC score is positive only if the Jacobian provides
    information BEYOND what activations reveal.

    Args:
        jac_diffs: np.array of Jacobian differences per prompt
        act_diffs: np.array of activation differences per prompt
        rho: Penalty weight on activation difference (default 1.0)

    Returns:
        np.array of NIAC scores per prompt
    """
    return jac_diffs - rho * act_diffs


def permutation_test(observed_stat, null_values, n_perms=10000, seed=42):
    """Permutation test for non-parametric p-value.

    Computes the fraction of permutations that produce a test statistic
    as extreme as the observed one.

    Args:
        observed_stat: The observed test statistic (scalar)
        null_values: np.array of values under H0 (will be permuted)
        n_perms: Number of permutations
        seed: Random seed

    Returns:
        dict with p_value, n_perms, observed_stat
    """
    rng = np.random.RandomState(seed)
    n = len(null_values)
    count_extreme = 0

    for _ in range(n_perms):
        perm = rng.choice(null_values, size=n, replace=True)
        perm_stat = np.mean(perm)
        if perm_stat >= observed_stat:
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_perms + 1)  # +1 for continuity correction

    return {
        "p_value": float(p_value),
        "n_perms": n_perms,
        "observed_stat": float(observed_stat),
    }


def temporal_accumulation(values):
    """Compute cumulative SNR as a function of observation count.

    Shows how detection confidence grows with more samples.

    Args:
        values: np.array of per-prompt differences

    Returns:
        dict with:
            n_obs: list of observation counts [1, 2, ..., N]
            cum_snr: list of cumulative SNR at each count
            cum_t: list of cumulative t-statistic at each count
    """
    n = len(values)
    n_obs = []
    cum_snr = []
    cum_t = []

    for k in range(2, n + 1):  # Need at least 2 for std
        subset = values[:k]
        mean_k = np.mean(subset)
        std_k = np.std(subset, ddof=1)
        se_k = std_k / np.sqrt(k)

        snr_k = mean_k / std_k if std_k > 1e-15 else float('inf')
        t_k = mean_k / se_k if se_k > 1e-15 else float('inf')

        n_obs.append(k)
        cum_snr.append(float(snr_k))
        cum_t.append(float(t_k))

    return {
        "n_obs": n_obs,
        "cum_snr": cum_snr,
        "cum_t": cum_t,
    }


def statistical_power(effect_size, n, alpha=0.01):
    """Compute statistical power for one-sample t-test.

    Power = P(reject H0 | H1 is true) at significance level alpha.

    Args:
        effect_size: Cohen's d = mean / std
        n: Sample size
        alpha: Significance level

    Returns:
        float: Statistical power
    """
    t_crit = stats.t.ppf(1 - alpha, df=n - 1)
    ncp = effect_size * np.sqrt(n)  # Non-centrality parameter
    power = 1.0 - stats.nct.cdf(t_crit, df=n - 1, nc=ncp)
    return float(power)


# ── Comprehensive analysis ──

def analyze_condition(act_diffs, jac_diffs, condition_name=""):
    """Run all statistical tests for a single experimental condition.

    Args:
        act_diffs: np.array of per-prompt activation differences
        jac_diffs: np.array of per-prompt Jacobian differences
        condition_name: Label for logging

    Returns:
        dict with all test results
    """
    n = len(act_diffs)

    # 1. One-sample t-tests
    act_ttest = one_sample_ttest(act_diffs)
    jac_ttest = one_sample_ttest(jac_diffs)

    # 2. NIAC Score
    niac_vals = niac_score(jac_diffs, act_diffs, rho=1.0)
    niac_ttest = one_sample_ttest(niac_vals)

    # 3. AUC-ROC (using zeros as clean baseline)
    auc_act = compute_auc(np.zeros(n), act_diffs)
    auc_jac = compute_auc(np.zeros(n), jac_diffs)
    auc_niac = compute_auc(np.zeros(n), niac_vals)

    # 4. F-test: is Jacobian variance inflated relative to activation variance?
    ftest = f_test_variance(jac_diffs, act_diffs)

    # 5. Temporal accumulation
    jac_temporal = temporal_accumulation(jac_diffs)
    act_temporal = temporal_accumulation(act_diffs)

    # 6. Permutation test on NIAC score
    niac_perm = permutation_test(
        np.mean(niac_vals),
        niac_vals - np.mean(niac_vals),  # Center under H0
        n_perms=10000,
    )

    # 7. Power analysis
    if act_ttest["std"] > 0:
        act_d = act_ttest["mean"] / act_ttest["std"]
    else:
        act_d = 0.0
    if jac_ttest["std"] > 0:
        jac_d = jac_ttest["mean"] / jac_ttest["std"]
    else:
        jac_d = 0.0

    act_power = statistical_power(act_d, n)
    jac_power = statistical_power(jac_d, n)

    # 8. Detection decision based on NIAC criteria
    jac_detects = jac_ttest["p_value"] < 0.001 and auc_jac > 0.95
    act_blind = auc_act < 0.7
    niac_positive = niac_ttest["p_value"] < 0.01

    if jac_detects and act_blind:
        decision = "NIAC-DETECTED"
    elif jac_detects and not act_blind:
        decision = "BOTH-DETECT"
    elif not jac_detects and act_blind:
        decision = "NEITHER-DETECTS"
    else:
        decision = "ACT-ONLY"

    return {
        "condition": condition_name,
        "n": n,
        # Activation stats
        "act_mean": float(np.mean(act_diffs)),
        "act_std": float(np.std(act_diffs, ddof=1)),
        "act_ttest_t": act_ttest["t_stat"],
        "act_ttest_p": act_ttest["p_value"],
        "act_power": act_power,
        "auc_act": auc_act,
        # Jacobian stats
        "jac_mean": float(np.mean(jac_diffs)),
        "jac_std": float(np.std(jac_diffs, ddof=1)),
        "jac_ttest_t": jac_ttest["t_stat"],
        "jac_ttest_p": jac_ttest["p_value"],
        "jac_power": jac_power,
        "auc_jac": auc_jac,
        # NIAC stats
        "niac_mean": float(np.mean(niac_vals)),
        "niac_std": float(np.std(niac_vals, ddof=1)),
        "niac_ttest_t": niac_ttest["t_stat"],
        "niac_ttest_p": niac_ttest["p_value"],
        "auc_niac": auc_niac,
        "niac_perm_p": niac_perm["p_value"],
        # F-test
        "f_stat": ftest["f_stat"],
        "f_pvalue": ftest["p_value"],
        # Temporal
        "jac_temporal": jac_temporal,
        "act_temporal": act_temporal,
        # Decision
        "decision": decision,
    }


def format_results_table(results_list):
    """Format a list of condition results as a readable table."""
    lines = []
    lines.append(f"\n{'='*100}")
    lines.append(f"{'Condition':<30} | {'act_mean':>9} | {'jac_mean':>9} | "
                 f"{'AUC_act':>8} | {'AUC_jac':>8} | {'jac_p':>10} | "
                 f"{'NIAC_p':>10} | {'Decision':<18}")
    lines.append(f"{'-'*30}-+-{'-'*9}-+-{'-'*9}-+-"
                 f"{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-"
                 f"{'-'*10}-+-{'-'*18}")

    for r in results_list:
        lines.append(
            f"{r['condition']:<30} | {r['act_mean']:>9.6f} | {r['jac_mean']:>9.6f} | "
            f"{r['auc_act']:>8.4f} | {r['auc_jac']:>8.4f} | {r['jac_ttest_p']:>10.2e} | "
            f"{r['niac_ttest_p']:>10.2e} | {r['decision']:<18}"
        )

    lines.append(f"{'='*100}")
    return "\n".join(lines)


def format_detailed_report(result):
    """Format detailed report for a single condition."""
    lines = []
    lines.append(f"\n--- {result['condition']} (n={result['n']}) ---")
    lines.append(f"  Activation: mean={result['act_mean']:.6f}, std={result['act_std']:.6f}")
    lines.append(f"    t={result['act_ttest_t']:.3f}, p={result['act_ttest_p']:.2e}, "
                 f"power={result['act_power']:.3f}, AUC={result['auc_act']:.4f}")
    lines.append(f"  Jacobian:   mean={result['jac_mean']:.6f}, std={result['jac_std']:.6f}")
    lines.append(f"    t={result['jac_ttest_t']:.3f}, p={result['jac_ttest_p']:.2e}, "
                 f"power={result['jac_power']:.3f}, AUC={result['auc_jac']:.4f}")
    lines.append(f"  NIAC Score: mean={result['niac_mean']:.6f}, std={result['niac_std']:.6f}")
    lines.append(f"    t={result['niac_ttest_t']:.3f}, p={result['niac_ttest_p']:.2e}, "
                 f"perm_p={result['niac_perm_p']:.4f}, AUC={result['auc_niac']:.4f}")
    lines.append(f"  F-test: F={result['f_stat']:.3f}, p={result['f_pvalue']:.2e}")
    lines.append(f"  Decision: {result['decision']}")
    return "\n".join(lines)


# ── v2: Separate calibration vs test analysis ──

def analyze_condition_v2(cal_act_diffs, cal_jac_diffs,
                         test_act_diffs, test_jac_diffs,
                         condition_name,
                         noise_floor_act=0.0, noise_floor_jac=0.0):
    """Analyze a condition with SEPARATE calibration and test data.

    This is the key improvement over v1: we measure Jacobian changes on
    calibration data where null-space guarantees act_diff ≈ 0.

    The critical comparison:
      - Cal data, null-space: act_diff ≈ noise_floor, jac_diff >> noise_floor
      - Cal data, gaussian:   act_diff >> noise_floor, jac_diff >> noise_floor

    Args:
        cal_act_diffs: np.array of per-prompt activation diffs on cal data
        cal_jac_diffs: np.array of per-prompt Jacobian diffs on cal data
        test_act_diffs: np.array of per-prompt activation diffs on test data
        test_jac_diffs: np.array of per-prompt Jacobian diffs on test data
        condition_name: Label
        noise_floor_act: Noise floor for activation (0 if deterministic)
        noise_floor_jac: Noise floor for Jacobian (0 if deterministic)

    Returns:
        dict with all analysis results
    """
    n_cal = len(cal_act_diffs)
    n_test = len(test_act_diffs)

    # ── Calibration data statistics ──
    cal_act_ttest = one_sample_ttest(cal_act_diffs)
    cal_jac_ttest = one_sample_ttest(cal_jac_diffs)
    cal_niac = niac_score(cal_jac_diffs, cal_act_diffs)
    cal_niac_ttest = one_sample_ttest(cal_niac)

    cal_act_mean = float(np.mean(cal_act_diffs))
    cal_jac_mean = float(np.mean(cal_jac_diffs))
    cal_jac_act_ratio = cal_jac_mean / (cal_act_mean + 1e-10)

    # ── Test data statistics ──
    test_act_ttest = one_sample_ttest(test_act_diffs)
    test_jac_ttest = one_sample_ttest(test_jac_diffs)
    test_niac = niac_score(test_jac_diffs, test_act_diffs)
    test_niac_ttest = one_sample_ttest(test_niac)

    test_act_mean = float(np.mean(test_act_diffs))
    test_jac_mean = float(np.mean(test_jac_diffs))
    test_jac_act_ratio = test_jac_mean / (test_act_mean + 1e-10)

    # ── AUC: using zeros as clean baseline ──
    # NOTE: When noise_floor=0 (deterministic forward pass), clean_scores are
    # all exactly 0.0 while perturbed_scores are all > 0, so AUC is trivially
    # 1.0 for ANY perturbation that produces nonzero diffs. This metric is only
    # informative when noise_floor > 0 or when comparing across conditions with
    # a shared, non-trivial clean baseline. Decision logic should NOT rely on
    # these AUC values — use t-test and NAF instead.
    auc_cal_act = compute_auc(np.zeros(n_cal), cal_act_diffs)
    auc_cal_jac = compute_auc(np.zeros(n_cal), cal_jac_diffs)
    auc_test_act = compute_auc(np.zeros(n_test), test_act_diffs)
    auc_test_jac = compute_auc(np.zeros(n_test), test_jac_diffs)

    # ── Effect size (Cohen's d) ──
    cal_act_d = cal_act_mean / (float(np.std(cal_act_diffs, ddof=1)) + 1e-10)
    cal_jac_d = cal_jac_mean / (float(np.std(cal_jac_diffs, ddof=1)) + 1e-10)

    return {
        "condition": condition_name,
        "n_cal": n_cal,
        "n_test": n_test,
        # Calibration data
        "cal_act_mean": cal_act_mean,
        "cal_act_std": float(np.std(cal_act_diffs, ddof=1)),
        "cal_act_ttest_t": cal_act_ttest["t_stat"],
        "cal_act_ttest_p": cal_act_ttest["p_value"],
        "cal_jac_mean": cal_jac_mean,
        "cal_jac_std": float(np.std(cal_jac_diffs, ddof=1)),
        "cal_jac_ttest_t": cal_jac_ttest["t_stat"],
        "cal_jac_ttest_p": cal_jac_ttest["p_value"],
        "cal_jac_act_ratio": cal_jac_act_ratio,
        "auc_cal_act": auc_cal_act,
        "auc_cal_jac": auc_cal_jac,
        "cal_niac_mean": float(np.mean(cal_niac)),
        "cal_niac_ttest_p": cal_niac_ttest["p_value"],
        "cal_act_cohens_d": cal_act_d,
        "cal_jac_cohens_d": cal_jac_d,
        # Test data
        "test_act_mean": test_act_mean,
        "test_act_std": float(np.std(test_act_diffs, ddof=1)),
        "test_act_ttest_t": test_act_ttest["t_stat"],
        "test_act_ttest_p": test_act_ttest["p_value"],
        "test_jac_mean": test_jac_mean,
        "test_jac_std": float(np.std(test_jac_diffs, ddof=1)),
        "test_jac_ttest_t": test_jac_ttest["t_stat"],
        "test_jac_ttest_p": test_jac_ttest["p_value"],
        "test_jac_act_ratio": test_jac_act_ratio,
        "auc_test_act": auc_test_act,
        "auc_test_jac": auc_test_jac,
        "test_niac_mean": float(np.mean(test_niac)),
        "test_niac_ttest_p": test_niac_ttest["p_value"],
        # Noise floor
        "noise_floor_act": noise_floor_act,
        "noise_floor_jac": noise_floor_jac,
    }


def format_results_table_v2(results_list):
    """Format v2 results with separate cal/test columns."""
    lines = []
    lines.append(f"\n{'='*130}")
    lines.append(
        f"{'Condition':<28} | {'cal_act':>9} | {'cal_jac':>9} | {'cal_j/a':>7} | "
        f"{'test_act':>9} | {'test_jac':>9} | {'test_j/a':>8} | "
        f"{'cal_jac_p':>10} | {'AUC_cj':>7}"
    )
    lines.append(
        f"{'-'*28}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}-+-"
        f"{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-"
        f"{'-'*10}-+-{'-'*7}"
    )

    for r in results_list:
        lines.append(
            f"{r['condition']:<28} | {r['cal_act_mean']:>9.6f} | {r['cal_jac_mean']:>9.6f} | "
            f"{r['cal_jac_act_ratio']:>7.1f}x | "
            f"{r['test_act_mean']:>9.6f} | {r['test_jac_mean']:>9.6f} | "
            f"{r['test_jac_act_ratio']:>7.1f}x | "
            f"{r['cal_jac_ttest_p']:>10.2e} | {r['auc_cal_jac']:>7.4f}"
        )

    lines.append(f"{'='*130}")
    return "\n".join(lines)
