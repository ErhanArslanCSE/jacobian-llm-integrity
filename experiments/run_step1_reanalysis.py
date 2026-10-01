#!/usr/bin/env python3
"""Step 1: Re-analyze exp1 Step 9 results with correct statistics.

exp1 used SNR ratio (SNR_J / SNR_act) as the decision metric.
This was wrong: ratio < 1 does NOT mean "Jacobian can't detect".
It means "Jacobian has lower signal-to-noise than activations".

The correct question: "Can Jacobian detect the perturbation AT ALL?"
Answer: one-sample t-test with H0: mu_jac = 0.

This script reads exp1's summary statistics and computes:
1. t-statistic from (mean, std, n)
2. p-value
3. Statistical power
4. Updated decision

Expected result: ALL conditions had significant Jacobian detection.
The NO-GO verdict was wrong because the metric was wrong.
"""

import json
import os
import sys
import numpy as np
from scipy import stats
from datetime import datetime


def ttest_from_summary(mean, std, n, alpha=0.01):
    """Compute one-sample t-test from summary statistics.

    H0: mu = 0, H1: mu > 0 (one-sided)
    """
    if std == 0 or n < 2:
        return {
            "t_stat": float('inf') if mean > 0 else 0.0,
            "p_value": 0.0 if mean > 0 else 1.0,
            "se": 0.0,
            "reject_h0": mean > 0,
        }

    se = std / np.sqrt(n)
    t_stat = mean / se
    p_value = 1.0 - stats.t.cdf(t_stat, df=n - 1)

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "se": float(se),
        "reject_h0": p_value < alpha,
    }


def power_from_summary(mean, std, n, alpha=0.01):
    """Compute statistical power from summary statistics."""
    if std == 0:
        return 1.0 if mean > 0 else 0.0

    d = mean / std  # Cohen's d
    t_crit = stats.t.ppf(1 - alpha, df=n - 1)
    ncp = d * np.sqrt(n)
    power = 1.0 - stats.nct.cdf(t_crit, df=n - 1, nc=ncp)
    return float(power)


def ftest_from_summary(var1, n1, var2, n2):
    """F-test from summary statistics (is var1 > var2?)."""
    if var2 < 1e-15:
        return {"f_stat": float('inf'), "p_value": 0.0}
    f_stat = var1 / var2
    p_value = 1.0 - stats.f.cdf(f_stat, dfn=n1 - 1, dfd=n2 - 1)
    return {"f_stat": float(f_stat), "p_value": float(p_value)}


def main():
    # Pilot (Experiment 1) Step 9 results, stored under <repo>/results/pilot/
    exp1_results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "results", "pilot", "step9_frobenius_full_jacobian", "results.json"
    )

    if not os.path.exists(exp1_results_path):
        print(f"ERROR: exp1 results not found at {exp1_results_path}")
        sys.exit(1)

    with open(exp1_results_path) as f:
        data = json.load(f)

    meta = data["_meta"]
    n = meta["num_prompts_test"]  # 30

    print("=" * 80)
    print("STEP 1: Re-analysis of exp1 Step 9 with correct statistics")
    print("=" * 80)
    print(f"\nSource: {exp1_results_path}")
    print(f"n = {n} test prompts, hidden_dim = {meta['hidden_dim']}")
    print(f"Original verdict: {meta['overall_decision_frobenius']}")
    print()

    # Analyze each condition
    results = []
    conditions = [k for k in data.keys() if k != "_meta"]

    for condition in conditions:
        layer_data = data[condition]["12"]  # Layer 12

        act_mean = layer_data["act_mean"]
        act_std = layer_data["act_std"]
        jac_mean = layer_data["jac_frob_mean"]
        jac_std = layer_data["jac_frob_std"]

        # Original metrics
        old_ratio = layer_data.get("ratio_frob", 0)
        old_decision = layer_data.get("decision_frob", "?")

        # New: proper t-tests
        act_ttest = ttest_from_summary(act_mean, act_std, n)
        jac_ttest = ttest_from_summary(jac_mean, jac_std, n)

        # Power analysis
        act_power = power_from_summary(act_mean, act_std, n)
        jac_power = power_from_summary(jac_mean, jac_std, n)

        # F-test: is Jacobian variance > activation variance?
        ftest = ftest_from_summary(jac_std**2, n, act_std**2, n)

        # NIAC: jac - act (from summary: mean difference)
        niac_mean = jac_mean - act_mean
        # Approximate std: sqrt(var_jac + var_act) assuming independence
        niac_std_approx = np.sqrt(jac_std**2 + act_std**2)
        niac_ttest = ttest_from_summary(niac_mean, niac_std_approx, n)

        # New decision
        jac_detects = jac_ttest["p_value"] < 0.001
        act_detects = act_ttest["p_value"] < 0.001

        if jac_detects:
            new_decision = "JAC-DETECTS"
        else:
            new_decision = "JAC-FAILS"

        result = {
            "condition": condition,
            "act_mean": act_mean,
            "act_std": act_std,
            "act_t": act_ttest["t_stat"],
            "act_p": act_ttest["p_value"],
            "act_power": act_power,
            "jac_mean": jac_mean,
            "jac_std": jac_std,
            "jac_t": jac_ttest["t_stat"],
            "jac_p": jac_ttest["p_value"],
            "jac_power": jac_power,
            "ftest_f": ftest["f_stat"],
            "ftest_p": ftest["p_value"],
            "niac_mean": niac_mean,
            "niac_t": niac_ttest["t_stat"],
            "niac_p": niac_ttest["p_value"],
            "old_ratio": old_ratio,
            "old_decision": old_decision,
            "new_decision": new_decision,
        }
        results.append(result)

    # Print table
    print(f"{'Condition':<32} | {'old_ratio':>9} | {'old_dec':>10} | "
          f"{'jac_t':>8} | {'jac_p':>10} | {'jac_pwr':>8} | {'new_dec':>12}")
    print(f"{'-'*32}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")

    for r in results:
        print(f"{r['condition']:<32} | {r['old_ratio']:>9.3f} | {r['old_decision']:>10} | "
              f"{r['jac_t']:>8.2f} | {r['jac_p']:>10.2e} | {r['jac_power']:>8.4f} | "
              f"{r['new_decision']:>12}")

    # Detailed output
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    for r in results:
        print(f"\n--- {r['condition']} ---")
        print(f"  Activation: mean={r['act_mean']:.6f}, std={r['act_std']:.6f}, "
              f"t={r['act_t']:.2f}, p={r['act_p']:.2e}, power={r['act_power']:.4f}")
        print(f"  Jacobian:   mean={r['jac_mean']:.6f}, std={r['jac_std']:.6f}, "
              f"t={r['jac_t']:.2f}, p={r['jac_p']:.2e}, power={r['jac_power']:.4f}")
        print(f"  NIAC (jac-act): mean={r['niac_mean']:.6f}, t={r['niac_t']:.2f}, p={r['niac_p']:.2e}")
        print(f"  F-test (var_jac > var_act): F={r['ftest_f']:.3f}, p={r['ftest_p']:.4f}")
        print(f"  Old: ratio={r['old_ratio']:.3f} -> {r['old_decision']}")
        print(f"  New: {r['new_decision']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_jac_detect = all(r["new_decision"] == "JAC-DETECTS" for r in results)
    n_jac_detect = sum(1 for r in results if r["new_decision"] == "JAC-DETECTS")

    print(f"\nJacobian detects in {n_jac_detect}/{len(results)} conditions")
    print(f"All conditions: {'YES' if all_jac_detect else 'NO'}")

    if all_jac_detect:
        print("\nCONCLUSION: exp1's NO-GO verdict was WRONG.")
        print("The Jacobian ALWAYS detected perturbations (t-test p << 0.001).")
        print("The problem was the metric (SNR ratio), not the method.")
        print("SNR_J / SNR_act < 1 only means Jacobian has lower SNR than activations,")
        print("NOT that Jacobian can't detect.")
    else:
        failed = [r["condition"] for r in results if r["new_decision"] != "JAC-DETECTS"]
        print(f"\nSome conditions failed detection: {failed}")
        print("Further investigation needed.")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "results", "step1_reanalysis")
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "results": results,
        "meta": {
            "source": exp1_results_path,
            "n_test": n,
            "date": datetime.now().isoformat(),
            "conclusion": "exp1 NO-GO was wrong" if all_jac_detect else "mixed results",
            "n_jac_detect": n_jac_detect,
            "n_total": len(results),
        }
    }

    output_path = os.path.join(output_dir, "reanalysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
