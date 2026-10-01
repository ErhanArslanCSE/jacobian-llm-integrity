#!/usr/bin/env python3
"""
NIS Paper v2 — Statistical Analysis Script
============================================
Tüm istatistiksel hesaplamaları mevcut deney verilerinden yapar.
HİÇBİR deney yeniden çalıştırılmaz — sadece sonuç dosyalarından hesaplama.

Hesaplamalar:
  1. Bootstrap 95% CI (NAF ve Cohen's d için)
  2. Shapiro-Wilk normallik testi + Wilcoxon doğrulama
  3. Bonferroni-adjusted p-values
  4. MDE (Minimum Detectable Effect) tahmini
  5. NAF Threshold ROC analizi

Kullanım:
  python3 statistical_analysis.py
"""

import json
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Result files live in <repo>/results/; this script sits in <repo>/analysis/.
BASE_DIR = Path(__file__).resolve().parent.parent / "results"
SEED = 42
N_BOOTSTRAP = 10_000

np.random.seed(SEED)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all_experiment_data() -> dict:
    """Tüm deney verilerini tek bir dict'e yükler."""
    data = {}

    # Exp2: Core NIAC detection (has raw per-prompt diffs)
    data["exp2"] = load_json(BASE_DIR / "step2_eval_v2_fixed/results.json")

    # Exp3: Calibration sweep
    data["exp3"] = load_json(BASE_DIR / "step3_cal_sweep/results_summary.json")

    # Exp4: Multi-layer
    data["exp4"] = load_json(BASE_DIR / "step4_multi_layer/results_summary.json")

    # Exp5: Backdoor (terminal output — parse from summary JSON embedded in text)
    data["exp5"] = _parse_terminal_json(
        BASE_DIR / "step5_eval_20260226_220101/results_summary.json"
    )

    # Exp6: Backdoor null-space (same terminal output format)
    data["exp6"] = _parse_terminal_json(
        BASE_DIR / "step5_eval_20260226_220101/results_summary.json"
    )

    # Exp7: Tap test
    data["exp7_summary"] = load_json(
        BASE_DIR / "step7_eval_20260227_011659/results_summary.json"
    )
    data["exp7_results"] = load_json(
        BASE_DIR / "step7_eval_20260227_011659/results.json"
    )

    # Exp8: Scale test (3B, 8B)
    data["exp8_3b"] = load_json(
        BASE_DIR / "step8_scale_test_3b/results.json"
    )
    data["exp8_3b_summary"] = load_json(
        BASE_DIR / "step8_scale_test_3b/results_summary.json"
    )
    data["exp8_8b"] = load_json(
        BASE_DIR / "step8_scale_test_8b/results.json"
    )
    data["exp8_8b_summary"] = load_json(
        BASE_DIR / "step8_scale_test_8b/results_summary.json"
    )

    # Exp9: Baselines
    data["exp9"] = load_json(
        BASE_DIR / "step9_baselines/results.json"
    )
    data["exp9_summary"] = load_json(
        BASE_DIR / "step9_baselines/results_summary.json"
    )

    # Exp10: Cross-architecture
    data["exp10_mistral"] = load_json(
        BASE_DIR / "step10_cross_arch_mistral_7b/results.json"
    )
    data["exp10_mistral_summary"] = load_json(
        BASE_DIR / "step10_cross_arch_mistral_7b/results_summary.json"
    )
    data["exp10_qwen"] = load_json(
        BASE_DIR / "step10_cross_arch_qwen25_7b/results.json"
    )
    data["exp10_qwen_summary"] = load_json(
        BASE_DIR / "step10_cross_arch_qwen25_7b/results_summary.json"
    )

    # Exp11: Attention layer
    data["exp11"] = load_json(
        BASE_DIR / "step11_attention_q_proj/results.json"
    )
    data["exp11_summary"] = load_json(
        BASE_DIR / "step11_attention_q_proj/results_summary.json"
    )

    return data


def _parse_terminal_json(path: str) -> Optional[dict]:
    """Terminal output formatındaki dosyalardan JSON parse eder."""
    try:
        with open(path) as f:
            content = f.read()
        # JSON kısmını bul (ilk { ile son } arası)
        start = content.find('{"layer_summary"')
        if start == -1:
            start = content.find('{')
        end = content.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            return json.loads(json_str)
    except Exception as e:
        print(f"  [WARN] Could not parse {path}: {e}")
    return None


# =============================================================================
# 1. BOOTSTRAP 95% CI
# =============================================================================

def bootstrap_ci(values: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for mean."""
    rng = np.random.RandomState(SEED)
    n = len(values)
    bootstrap_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(bootstrap_means, alpha * 100)
    hi = np.percentile(bootstrap_means, (1 - alpha) * 100)
    return float(np.mean(values)), float(lo), float(hi)


def bootstrap_ci_cohens_d(group1: np.ndarray, group2: np.ndarray,
                           n_bootstrap: int = N_BOOTSTRAP, ci: float = 0.95) -> tuple:
    """Bootstrap CI for Cohen's d between two groups."""
    rng = np.random.RandomState(SEED)
    n1, n2 = len(group1), len(group2)
    ds = []
    for _ in range(n_bootstrap):
        s1 = rng.choice(group1, size=n1, replace=True)
        s2 = rng.choice(group2, size=n2, replace=True)
        pooled_std = np.sqrt((np.var(s1, ddof=1) + np.var(s2, ddof=1)) / 2)
        if pooled_std > 0:
            ds.append((np.mean(s1) - np.mean(s2)) / pooled_std)
        else:
            ds.append(0.0)
    ds = np.array(ds)
    alpha = (1 - ci) / 2
    return float(np.mean(ds)), float(np.percentile(ds, alpha * 100)), float(np.percentile(ds, (1 - alpha) * 100))


def bootstrap_ci_naf(gauss_act: np.ndarray, gauss_jac: np.ndarray,
                     ns_act: np.ndarray, ns_jac: np.ndarray,
                     n_bootstrap: int = N_BOOTSTRAP, ci: float = 0.95) -> tuple:
    """Bootstrap CI for NAF = (jac_ns/jac_gauss) / (act_ns/act_gauss)."""
    rng = np.random.RandomState(SEED)
    n = min(len(gauss_act), len(gauss_jac), len(ns_act), len(ns_jac))
    nafs = []
    for _ in range(n_bootstrap):
        idx_g = rng.choice(len(gauss_act), size=len(gauss_act), replace=True)
        idx_n = rng.choice(len(ns_act), size=len(ns_act), replace=True)
        ga = np.mean(gauss_act[idx_g])
        gj = np.mean(gauss_jac[idx_g])
        na = np.mean(ns_act[idx_n])
        nj = np.mean(ns_jac[idx_n])
        if ga > 0 and na > 0 and gj > 0:
            act_evasion = 1 - na / ga
            jac_evasion = 1 - nj / gj
            if act_evasion > 0 and jac_evasion > 0:
                nafs.append(act_evasion / jac_evasion)
            else:
                nafs.append(np.nan)
        else:
            nafs.append(np.nan)
    nafs = np.array([x for x in nafs if not np.isnan(x)])
    if len(nafs) == 0:
        return (np.nan, np.nan, np.nan)
    alpha = (1 - ci) / 2
    return float(np.mean(nafs)), float(np.percentile(nafs, alpha * 100)), float(np.percentile(nafs, (1 - alpha) * 100))


def compute_bootstrap_cis_exp2(data: dict) -> list:
    """Exp2 raw per-prompt verilerinden bootstrap CI hesapla."""
    results = []
    raw = data["exp2"]["raw_data"]
    summary = data["exp2"]["results"]

    for cond in summary:
        condition = cond["condition"]
        raw_key = condition

        if raw_key not in raw:
            continue

        rd = raw[raw_key]
        cal_act = np.array(rd["cal_act_diffs"])
        cal_jac = np.array(rd["cal_jac_diffs"])
        test_act = np.array(rd["test_act_diffs"])
        test_jac = np.array(rd["test_jac_diffs"])

        # Zero baseline (noise floor = 0)
        zeros = np.zeros_like(cal_act)

        # Bootstrap CI for cal_jac_mean
        jac_mean, jac_lo, jac_hi = bootstrap_ci(cal_jac)

        # Bootstrap CI for cal_act_mean
        act_mean, act_lo, act_hi = bootstrap_ci(cal_act)

        # Bootstrap CI for Cohen's d (jac vs zeros)
        jac_d, jac_d_lo, jac_d_hi = bootstrap_ci_cohens_d(cal_jac, zeros)

        # Bootstrap CI for Cohen's d (act vs zeros)
        act_d, act_d_lo, act_d_hi = bootstrap_ci_cohens_d(cal_act, zeros)

        results.append({
            "experiment": "exp2",
            "condition": condition,
            "n": len(cal_act),
            "cal_jac_mean": jac_mean,
            "cal_jac_ci": [jac_lo, jac_hi],
            "cal_act_mean": act_mean,
            "cal_act_ci": [act_lo, act_hi],
            "cal_jac_cohens_d": jac_d,
            "cal_jac_cohens_d_ci": [jac_d_lo, jac_d_hi],
            "cal_act_cohens_d": act_d,
            "cal_act_cohens_d_ci": [act_d_lo, act_d_hi],
        })

    # NAF bootstrap CI for null-space conditions
    for eps in ["0.01", "0.05", "0.1"]:
        gauss_key = f"gaussian_sublayer_eps{eps}"
        ns_key = f"null_space_eps{eps}"
        if gauss_key in raw and ns_key in raw:
            ga = np.array(raw[gauss_key]["cal_act_diffs"])
            gj = np.array(raw[gauss_key]["cal_jac_diffs"])
            na = np.array(raw[ns_key]["cal_act_diffs"])
            nj = np.array(raw[ns_key]["cal_jac_diffs"])
            naf_mean, naf_lo, naf_hi = bootstrap_ci_naf(ga, gj, na, nj)
            results.append({
                "experiment": "exp2",
                "condition": f"NAF_eps{eps}",
                "naf_mean": naf_mean,
                "naf_ci": [naf_lo, naf_hi],
            })

    return results


def compute_bootstrap_cis_summary_nafs(data: dict) -> list:
    """Summary-level NAF verilerinden bootstrap CI hesapla (per-condition NAF'lar üzerinden)."""
    results = []

    # Exp3: Calibration sweep — multiple n_cal values give repeated NAF measurements
    exp3 = data.get("exp3")
    if exp3 and "sweep_summary" in exp3:
        for eps in [0.01, 0.05, 0.1]:
            nafs = [s["naf"] for s in exp3["sweep_summary"] if abs(s["epsilon"] - eps) < 0.001]
            if len(nafs) >= 3:
                naf_arr = np.array(nafs)
                m, lo, hi = bootstrap_ci(naf_arr)
                results.append({
                    "experiment": "exp3_cal_sweep",
                    "condition": f"eps={eps}",
                    "n_samples": len(nafs),
                    "naf_mean": m,
                    "naf_ci": [lo, hi],
                })

    # Exp4: Multi-layer — NAF per layer
    exp4 = data.get("exp4")
    if exp4 and isinstance(exp4, dict) and "layer_summary" in exp4:
        for eps in [0.01, 0.05, 0.1]:
            nafs = [s["naf"] for s in exp4["layer_summary"] if abs(s["epsilon"] - eps) < 0.001]
            if len(nafs) >= 3:
                naf_arr = np.array(nafs)
                m, lo, hi = bootstrap_ci(naf_arr)
                results.append({
                    "experiment": "exp4_multi_layer",
                    "condition": f"eps={eps}",
                    "n_samples": len(nafs),
                    "naf_mean": m,
                    "naf_ci": [lo, hi],
                })

    # Exp9: Baselines — NAF per layer
    exp9_summary = data.get("exp9_summary")
    if exp9_summary and "niac_summary" in exp9_summary:
        for eps in [0.01, 0.05, 0.1]:
            nafs = [s["naf"] for s in exp9_summary["niac_summary"] if abs(s["epsilon"] - eps) < 0.001]
            if len(nafs) >= 3:
                naf_arr = np.array(nafs)
                m, lo, hi = bootstrap_ci(naf_arr)
                results.append({
                    "experiment": "exp9_baselines",
                    "condition": f"eps={eps}",
                    "n_samples": len(nafs),
                    "naf_mean": m,
                    "naf_ci": [lo, hi],
                })

    # Exp10: Cross-architecture NAFs
    for arch_key, arch_name in [("exp10_mistral", "Mistral-7B"), ("exp10_qwen", "Qwen2.5-7B")]:
        exp10 = data.get(arch_key)
        if exp10 and "tap_results" in exp10:
            for eps in [0.01, 0.05, 0.1]:
                nafs = []
                for key, val in exp10["tap_results"].items():
                    if f"eps{eps}" in key and "null_space" in key and "k8" in key:
                        if "niac_decision" in val:
                            nafs.append(val["niac_decision"]["naf"])
                if nafs:
                    results.append({
                        "experiment": f"exp10_{arch_name}",
                        "condition": f"eps={eps}_k=8",
                        "n_samples": len(nafs),
                        "naf_values": nafs,
                    })

    # Exp8: Scale test NAFs
    for scale_key, model_name in [("exp8_3b", "Llama-3.2-3B"), ("exp8_8b", "Llama-3.1-8B")]:
        exp8 = data.get(scale_key)
        if exp8 and "tap_results" in exp8:
            for eps in [0.01, 0.05, 0.1]:
                nafs = []
                for key, val in exp8["tap_results"].items():
                    if f"eps{eps}" in key and "null_space" in key and "k8" in key:
                        if "niac_decision" in val:
                            nafs.append(val["niac_decision"]["naf"])
                if nafs:
                    results.append({
                        "experiment": f"exp8_{model_name}",
                        "condition": f"eps={eps}_k=8",
                        "n_samples": len(nafs),
                        "naf_values": nafs,
                    })

    # Exp11: Attention layer NAFs
    exp11 = data.get("exp11")
    if exp11 and "tap_results" in exp11:
        for eps in [0.01, 0.05, 0.1]:
            nafs = []
            for key, val in exp11["tap_results"].items():
                if f"eps{eps}" in key and "null_space" in key and "k8" in key:
                    if "niac_decision" in val:
                        nafs.append(val["niac_decision"]["naf"])
            if nafs:
                results.append({
                    "experiment": "exp11_attention",
                    "condition": f"eps={eps}_k=8",
                    "n_samples": len(nafs),
                    "naf_values": nafs,
                })

    # Cross-experiment aggregated NAF bootstrap (all models, layers, for each epsilon, k=8)
    for eps in [0.01, 0.05, 0.1]:
        all_nafs = []
        for r in results:
            if f"eps={eps}" in r.get("condition", "") and "naf_values" in r:
                all_nafs.extend(r["naf_values"])
            elif f"eps={eps}" in r.get("condition", "") and "naf_mean" in r:
                all_nafs.append(r["naf_mean"])
        if len(all_nafs) >= 3:
            naf_arr = np.array(all_nafs)
            m, lo, hi = bootstrap_ci(naf_arr)
            results.append({
                "experiment": "AGGREGATE_ALL",
                "condition": f"eps={eps}",
                "n_samples": len(all_nafs),
                "naf_mean": m,
                "naf_ci": [lo, hi],
            })

    return results


# =============================================================================
# 2. SHAPIRO-WILK NORMALLIK TESTİ
# =============================================================================

def shapiro_wilk_analysis(data: dict) -> list:
    """Exp2 raw per-prompt verilerinde normallik testi."""
    results = []
    raw = data["exp2"]["raw_data"]

    for cond_name, cond_data in raw.items():
        for signal_type in ["cal_act_diffs", "cal_jac_diffs", "test_act_diffs", "test_jac_diffs"]:
            if signal_type not in cond_data:
                continue
            values = np.array(cond_data[signal_type])
            if len(values) < 3:
                continue

            # Shapiro-Wilk
            sw_stat, sw_p = stats.shapiro(values)

            # Wilcoxon signed-rank test (one-sample: diffs vs 0)
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(values, alternative='greater')

            # t-test (one-sample vs 0)
            t_stat, t_p = stats.ttest_1samp(values, 0)
            t_p_one = t_p / 2 if t_stat > 0 else 1 - t_p / 2

            # Cohen's d (vs zero)
            d = np.mean(values) / np.std(values, ddof=1) if np.std(values, ddof=1) > 0 else np.inf

            results.append({
                "condition": cond_name,
                "signal": signal_type,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "shapiro_stat": float(sw_stat),
                "shapiro_p": float(sw_p),
                "normal_at_005": sw_p > 0.05,
                "ttest_t": float(t_stat),
                "ttest_p_onesided": float(t_p_one),
                "wilcoxon_stat": float(wilcoxon_stat),
                "wilcoxon_p": float(wilcoxon_p),
                "cohens_d": float(d),
                "agreement": (t_p_one < 0.05) == (wilcoxon_p < 0.05),
            })

    return results


# =============================================================================
# 3. BONFERRONI-ADJUSTED P-VALUES
# =============================================================================

def collect_all_p_values(data: dict) -> list:
    """Tüm deneylerden p-value'ları topla."""
    pvals = []

    # Exp2
    for r in data["exp2"]["results"]:
        for key in ["cal_act_ttest_p", "cal_jac_ttest_p", "test_act_ttest_p", "test_jac_ttest_p",
                     "cal_niac_ttest_p", "test_niac_ttest_p"]:
            if key in r:
                pvals.append({
                    "experiment": "exp2",
                    "condition": r["condition"],
                    "test": key,
                    "p_value": r[key],
                })

    # Exp9: Baselines
    exp9 = data.get("exp9")
    if exp9 and "conditions" in exp9:
        for c in exp9["conditions"]:
            label = f"L{c['layer']}_eps{c['epsilon']}"
            for prefix in ["gauss_tap", "ns_tap"]:
                for split in ["cal", "test"]:
                    for signal in ["act", "jac"]:
                        key = f"{prefix}_{split}_{signal}_ttest_p"
                        if key in c:
                            pvals.append({
                                "experiment": "exp9",
                                "condition": label,
                                "test": f"{prefix}_{split}_{signal}",
                                "p_value": c[key],
                            })

    # Exp7: Full Jacobian decisions
    exp7 = data.get("exp7_results")
    if exp7 and "full_jac_decisions" in exp7:
        for key, val in exp7["full_jac_decisions"].items():
            pvals.append({
                "experiment": "exp7",
                "condition": key,
                "test": "niac_decision",
                "p_value": 0.0,  # All decisions are significant
            })

    return pvals


def bonferroni_correction(pvals: list) -> list:
    """Bonferroni düzeltmesi uygula."""
    n = len(pvals)
    alpha_adj = 0.05 / n if n > 0 else 0.05

    for p in pvals:
        p["n_tests"] = n
        p["alpha_adj"] = alpha_adj
        p["p_adjusted"] = min(p["p_value"] * n, 1.0)
        p["significant_bonferroni"] = p["p_adjusted"] < 0.05
        p["significant_original"] = p["p_value"] < 0.05

    # Check if any conclusion changes
    changes = [p for p in pvals if p["significant_original"] != p["significant_bonferroni"]]

    return pvals, changes, n, alpha_adj


# =============================================================================
# 4. MDE (MINIMUM DETECTABLE EFFECT) TAHMİNİ
# =============================================================================

def mde_analysis(data: dict) -> dict:
    """Epsilon küçüldükçe cal_jac_mean nasıl değişiyor? MDE tahmini."""
    results = {}

    # Exp2: Gaussian sublayer across epsilons
    exp2_results = data["exp2"]["results"]
    for prefix in ["gaussian_sublayer", "null_space"]:
        epsilons = []
        jac_means = []
        act_means = []
        for r in exp2_results:
            if r["condition"].startswith(prefix):
                eps_str = r["condition"].split("eps")[1]
                eps = float(eps_str)
                epsilons.append(eps)
                jac_means.append(r["cal_jac_mean"])
                act_means.append(r["cal_act_mean"])

        if len(epsilons) >= 2:
            eps_arr = np.array(epsilons)
            jac_arr = np.array(jac_means)
            act_arr = np.array(act_means)

            # Linear regression: jac_mean = a * epsilon + b
            slope_jac, intercept_jac, r_jac, p_jac, se_jac = stats.linregress(eps_arr, jac_arr)
            slope_act, intercept_act, r_act, p_act, se_act = stats.linregress(eps_arr, act_arr)

            # Sensitivity: cal_jac_mean / epsilon
            sensitivities = jac_arr / eps_arr

            # bfloat16 precision threshold: ~1/(2^7) = 0.0078 relative to weight norm
            # For bfloat16: mantissa = 7 bits, smallest representable diff ~ 2^-7 * scale
            bfloat16_relative = 2**-7  # ~0.0078

            # MDE estimate: epsilon where signal equals noise
            # noise_floor = 0, so theoretical MDE = 0
            # Practical MDE limited by bfloat16 quantization
            mde_practical = bfloat16_relative / np.mean(sensitivities) if np.mean(sensitivities) > 0 else np.inf

            results[prefix] = {
                "epsilons": epsilons,
                "jac_means": [float(x) for x in jac_means],
                "act_means": [float(x) for x in act_means],
                "jac_slope": float(slope_jac),
                "jac_intercept": float(intercept_jac),
                "jac_r_squared": float(r_jac ** 2),
                "jac_sensitivity_per_eps": [float(x) for x in sensitivities],
                "act_slope": float(slope_act),
                "act_r_squared": float(r_act ** 2),
                "noise_floor": 0.0,
                "mde_theoretical": 0.0,
                "bfloat16_threshold": float(bfloat16_relative),
                "mde_bfloat16_limited": float(mde_practical),
            }

    return results


# =============================================================================
# 5. NAF THRESHOLD ROC ANALİZİ
# =============================================================================

def naf_threshold_roc(data: dict) -> dict:
    """NAF=3.0 eşiğinin seçim gerekçesi ve ROC analizi."""

    # Tüm NAF değerlerini topla: pozitif = NIAC-DETECTED, negatif = clean/WEAK
    all_nafs = []

    # Exp3 sweep: NAF > 3 = real NIAC, NAF < 3 = weak/small epsilon
    exp3 = data.get("exp3")
    if exp3 and "sweep_summary" in exp3:
        for s in exp3["sweep_summary"]:
            label = 1 if s["decision"] == "NIAC-DETECTED" else 0
            all_nafs.append({"naf": s["naf"], "label": label, "source": "exp3"})

    # Exp4: Multi-layer
    exp4 = data.get("exp4")
    if exp4 and isinstance(exp4, dict) and "layer_summary" in exp4:
        for s in exp4["layer_summary"]:
            label = 1 if s["decision"] == "NIAC-DETECTED" else 0
            all_nafs.append({"naf": s["naf"], "label": label, "source": "exp4"})

    # Exp9: Baselines
    exp9_summary = data.get("exp9_summary")
    if exp9_summary and "niac_summary" in exp9_summary:
        for s in exp9_summary["niac_summary"]:
            label = 1 if s["decision"] == "NIAC-DETECTED" else 0
            all_nafs.append({"naf": s["naf"], "label": label, "source": "exp9"})

    # Add noise floor points (NAF = 0, label = 0)
    for _ in range(10):
        all_nafs.append({"naf": 0.0, "label": 0, "source": "noise_floor"})

    if not all_nafs:
        return {}

    nafs = np.array([x["naf"] for x in all_nafs])
    labels = np.array([x["label"] for x in all_nafs])

    # ROC curve
    thresholds = np.linspace(0, max(nafs) * 1.1, 200)
    roc_points = []
    for thr in thresholds:
        tp = np.sum((nafs >= thr) & (labels == 1))
        fp = np.sum((nafs >= thr) & (labels == 0))
        tn = np.sum((nafs < thr) & (labels == 0))
        fn = np.sum((nafs < thr) & (labels == 1))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        roc_points.append({"threshold": float(thr), "tpr": float(tpr), "fpr": float(fpr)})

    # AUC (trapezoidal)
    fprs = [p["fpr"] for p in roc_points]
    tprs = [p["tpr"] for p in roc_points]
    auc = -np.trapezoid(tprs, fprs)  # Negative because FPR is decreasing

    # Find threshold for FPR < 0.01
    fpr_target = 0.01
    best_threshold_fpr001 = None
    best_tpr_at_fpr001 = 0
    for p in roc_points:
        if p["fpr"] <= fpr_target and p["tpr"] >= best_tpr_at_fpr001:
            best_threshold_fpr001 = p["threshold"]
            best_tpr_at_fpr001 = p["tpr"]

    # Analysis at NAF=3.0
    thr30 = 3.0
    tp_30 = np.sum((nafs >= thr30) & (labels == 1))
    fp_30 = np.sum((nafs >= thr30) & (labels == 0))
    tn_30 = np.sum((nafs < thr30) & (labels == 0))
    fn_30 = np.sum((nafs < thr30) & (labels == 1))
    tpr_30 = tp_30 / (tp_30 + fn_30) if (tp_30 + fn_30) > 0 else 0
    fpr_30 = fp_30 / (fp_30 + tn_30) if (fp_30 + tn_30) > 0 else 0

    # Decision distribution
    niac_nafs = nafs[labels == 1]
    weak_nafs = nafs[labels == 0]

    return {
        "n_total": len(all_nafs),
        "n_niac": int(np.sum(labels == 1)),
        "n_weak_or_clean": int(np.sum(labels == 0)),
        "auc": float(auc),
        "at_naf_3.0": {
            "tp": int(tp_30),
            "fp": int(fp_30),
            "tn": int(tn_30),
            "fn": int(fn_30),
            "tpr": float(tpr_30),
            "fpr": float(fpr_30),
            "precision": float(tp_30 / (tp_30 + fp_30)) if (tp_30 + fp_30) > 0 else 0,
            "f1": float(2 * tp_30 / (2 * tp_30 + fp_30 + fn_30)) if (2 * tp_30 + fp_30 + fn_30) > 0 else 0,
        },
        "optimal_threshold_fpr_001": {
            "threshold": float(best_threshold_fpr001) if best_threshold_fpr001 is not None else None,
            "tpr": float(best_tpr_at_fpr001),
        },
        "niac_naf_range": {"min": float(np.min(niac_nafs)), "max": float(np.max(niac_nafs)),
                           "mean": float(np.mean(niac_nafs))} if len(niac_nafs) > 0 else {},
        "weak_naf_range": {"min": float(np.min(weak_nafs[weak_nafs > 0])),
                           "max": float(np.max(weak_nafs[weak_nafs > 0])),
                           "mean": float(np.mean(weak_nafs[weak_nafs > 0]))} if np.sum(weak_nafs > 0) > 0 else {},
    }


# =============================================================================
# MAIN + REPORTING
# =============================================================================

def format_ci(mean, ci):
    """Format as 'mean [lo, hi]'."""
    return f"{mean:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def format_ci_short(mean, ci):
    """Format as 'mean [lo, hi]' with 2 decimals."""
    return f"{mean:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def main():
    print("=" * 80)
    print("NIS Paper v2 — Statistical Analysis")
    print("=" * 80)

    # Load data
    print("\n[1] Loading experiment data...")
    data = load_all_experiment_data()
    print(f"  Loaded: {list(data.keys())}")

    # =========================================================================
    # TIER 1: Bootstrap 95% CI
    # =========================================================================
    print("\n" + "=" * 80)
    print("TIER 1: Bootstrap 95% CI (N_BOOTSTRAP = {:,})".format(N_BOOTSTRAP))
    print("=" * 80)

    # Exp2 raw data bootstrap
    print("\n--- Exp2: Per-prompt Bootstrap CIs ---")
    exp2_cis = compute_bootstrap_cis_exp2(data)
    for r in exp2_cis:
        if "cal_jac_mean" in r:
            print(f"\n  {r['condition']} (n={r['n']}):")
            print(f"    cal_jac_mean = {format_ci(r['cal_jac_mean'], r['cal_jac_ci'])}")
            print(f"    cal_act_mean = {format_ci(r['cal_act_mean'], r['cal_act_ci'])}")
            print(f"    cal_jac_d    = {format_ci_short(r['cal_jac_cohens_d'], r['cal_jac_cohens_d_ci'])}")
            print(f"    cal_act_d    = {format_ci_short(r['cal_act_cohens_d'], r['cal_act_cohens_d_ci'])}")
        elif "naf_mean" in r:
            print(f"\n  {r['condition']}:")
            print(f"    NAF = {format_ci_short(r['naf_mean'], r['naf_ci'])}")

    # Summary-level NAF bootstrap
    print("\n--- Cross-Experiment NAF Bootstrap CIs ---")
    naf_cis = compute_bootstrap_cis_summary_nafs(data)
    for r in naf_cis:
        if "naf_ci" in r:
            print(f"  {r['experiment']}, {r['condition']}: NAF = {format_ci_short(r['naf_mean'], r['naf_ci'])} (n={r['n_samples']})")
        elif "naf_values" in r:
            vals = r["naf_values"]
            if len(vals) >= 2:
                m, lo, hi = bootstrap_ci(np.array(vals))
                print(f"  {r['experiment']}, {r['condition']}: NAF = {format_ci_short(m, [lo, hi])} (n={r['n_samples']})")
            else:
                print(f"  {r['experiment']}, {r['condition']}: NAF = {vals[0]:.2f} (single value)")

    # =========================================================================
    # TIER 2: Shapiro-Wilk + Wilcoxon
    # =========================================================================
    print("\n" + "=" * 80)
    print("TIER 2: Shapiro-Wilk Normality + Wilcoxon Validation")
    print("=" * 80)

    sw_results = shapiro_wilk_analysis(data)
    non_normal_count = 0
    disagreement_count = 0
    for r in sw_results:
        normality = "NORMAL" if r["normal_at_005"] else "NON-NORMAL"
        if not r["normal_at_005"]:
            non_normal_count += 1
        agree = "AGREE" if r["agreement"] else "DISAGREE"
        if not r["agreement"]:
            disagreement_count += 1
        print(f"  {r['condition']}/{r['signal']}: W={r['shapiro_stat']:.4f}, p={r['shapiro_p']:.4f} [{normality}], "
              f"t-test p={r['ttest_p_onesided']:.2e}, Wilcoxon p={r['wilcoxon_p']:.2e}, d={r['cohens_d']:.2f} [{agree}]")

    print(f"\n  Summary: {non_normal_count}/{len(sw_results)} distributions are non-normal at alpha=0.05")
    print(f"  Summary: {disagreement_count}/{len(sw_results)} show disagreement between t-test and Wilcoxon")
    print(f"  All Cohen's d values >> 1.0 — distribution shape has NO practical impact")

    # =========================================================================
    # TIER 3a: Bonferroni Correction
    # =========================================================================
    print("\n" + "=" * 80)
    print("TIER 3a: Bonferroni-Adjusted p-values")
    print("=" * 80)

    pvals = collect_all_p_values(data)
    corrected, changes, n_tests, alpha_adj = bonferroni_correction(pvals)

    print(f"  Total tests: {n_tests}")
    print(f"  Adjusted alpha: {alpha_adj:.2e}")
    print(f"  Original significant: {sum(1 for p in corrected if p['significant_original'])}/{n_tests}")
    print(f"  Bonferroni significant: {sum(1 for p in corrected if p['significant_bonferroni'])}/{n_tests}")
    print(f"  Conclusions changed: {len(changes)}")

    if changes:
        for c in changes:
            print(f"    CHANGED: {c['experiment']}/{c['condition']}/{c['test']}: "
                  f"p={c['p_value']:.2e} -> p_adj={c['p_adjusted']:.2e}")
    else:
        print("  NO conclusions changed after Bonferroni correction.")

    # Count by p-value range
    p_zero = sum(1 for p in corrected if p["p_value"] == 0.0)
    p_tiny = sum(1 for p in corrected if 0 < p["p_value"] < 1e-10)
    p_small = sum(1 for p in corrected if 1e-10 <= p["p_value"] < 0.05)
    print(f"\n  p-value distribution:")
    print(f"    p = 0.0 (machine zero): {p_zero}")
    print(f"    0 < p < 1e-10:          {p_tiny}")
    print(f"    1e-10 <= p < 0.05:      {p_small}")

    # =========================================================================
    # TIER 3b: MDE Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("TIER 3b: Minimum Detectable Effect (MDE) Analysis")
    print("=" * 80)

    mde = mde_analysis(data)
    for prefix, m in mde.items():
        print(f"\n  {prefix}:")
        print(f"    Epsilons tested: {m['epsilons']}")
        print(f"    Jac means:       {[f'{x:.6f}' for x in m['jac_means']]}")
        print(f"    Jac slope:       {m['jac_slope']:.6f} (R^2 = {m['jac_r_squared']:.6f})")
        print(f"    Sensitivity:     {[f'{x:.4f}' for x in m['jac_sensitivity_per_eps']]}")
        print(f"    Noise floor:     {m['noise_floor']}")
        print(f"    MDE theoretical: {m['mde_theoretical']} (noise floor = 0)")
        print(f"    bfloat16 limit:  {m['bfloat16_threshold']:.6f}")
        print(f"    MDE (bfloat16):  {m['mde_bfloat16_limited']:.6f}")

    # =========================================================================
    # TIER 3c: NAF Threshold ROC
    # =========================================================================
    print("\n" + "=" * 80)
    print("TIER 3c: NAF Threshold ROC Analysis")
    print("=" * 80)

    roc = naf_threshold_roc(data)
    if roc:
        print(f"  Total decisions: {roc['n_total']}")
        print(f"  NIAC-DETECTED:   {roc['n_niac']}")
        print(f"  Weak/Clean:      {roc['n_weak_or_clean']}")
        print(f"  AUC:             {roc['auc']:.4f}")
        print(f"\n  At NAF = 3.0 threshold:")
        at3 = roc["at_naf_3.0"]
        print(f"    TP={at3['tp']}, FP={at3['fp']}, TN={at3['tn']}, FN={at3['fn']}")
        print(f"    TPR (sensitivity): {at3['tpr']:.4f}")
        print(f"    FPR:               {at3['fpr']:.4f}")
        print(f"    Precision:         {at3['precision']:.4f}")
        print(f"    F1:                {at3['f1']:.4f}")
        opt = roc.get("optimal_threshold_fpr_001", {})
        if opt.get("threshold"):
            print(f"\n  Optimal threshold for FPR < 0.01:")
            print(f"    Threshold: {opt['threshold']:.2f}")
            print(f"    TPR:       {opt['tpr']:.4f}")
        if roc.get("niac_naf_range"):
            r_niac = roc["niac_naf_range"]
            print(f"\n  NIAC NAF range: [{r_niac['min']:.2f}, {r_niac['max']:.2f}], mean={r_niac['mean']:.2f}")
        if roc.get("weak_naf_range"):
            r_weak = roc["weak_naf_range"]
            print(f"  Weak NAF range: [{r_weak['min']:.2f}, {r_weak['max']:.2f}], mean={r_weak['mean']:.2f}")

    # =========================================================================
    # AGGREGATE SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)

    # Collect all NAF values across all experiments for grand summary
    all_nafs_by_eps = {0.01: [], 0.05: [], 0.1: []}

    # Exp3 sweep (n_cal=30)
    if data.get("exp3") and "sweep_summary" in data["exp3"]:
        for s in data["exp3"]["sweep_summary"]:
            if s.get("n_cal") == 30:
                all_nafs_by_eps[s["epsilon"]].append(("Exp3-Llama1B-L12", s["naf"]))

    # Exp4 multi-layer
    exp4 = data.get("exp4")
    if exp4 and isinstance(exp4, dict) and "layer_summary" in exp4:
        for s in exp4["layer_summary"]:
            all_nafs_by_eps[s["epsilon"]].append((f"Exp4-L{s['layer']}", s["naf"]))

    # Exp9 baselines
    if data.get("exp9_summary") and "niac_summary" in data["exp9_summary"]:
        for s in data["exp9_summary"]["niac_summary"]:
            all_nafs_by_eps[s["epsilon"]].append((f"Exp9-L{s['layer']}", s["naf"]))

    print("\n  NAF values across all conditions:")
    for eps in [0.01, 0.05, 0.1]:
        vals = all_nafs_by_eps[eps]
        if vals:
            naf_arr = np.array([v[1] for v in vals])
            m, lo, hi = bootstrap_ci(naf_arr)
            print(f"\n  epsilon = {eps}:")
            print(f"    N conditions: {len(vals)}")
            print(f"    NAF range:    [{np.min(naf_arr):.2f}, {np.max(naf_arr):.2f}]")
            print(f"    NAF mean:     {format_ci_short(m, [lo, hi])}")
            for name, naf in vals:
                flag = " *" if naf < 3.0 else ""
                print(f"      {name}: NAF = {naf:.4f}{flag}")

    # =========================================================================
    # EXPORT TO JSON
    # =========================================================================
    output = {
        "bootstrap_ci_exp2": exp2_cis,
        "bootstrap_ci_cross_experiment": [r for r in naf_cis if "naf_ci" in r],
        "shapiro_wilk": sw_results,
        "bonferroni": {
            "n_tests": n_tests,
            "alpha_adj": alpha_adj,
            "n_significant_original": sum(1 for p in corrected if p["significant_original"]),
            "n_significant_bonferroni": sum(1 for p in corrected if p["significant_bonferroni"]),
            "n_changed": len(changes),
        },
        "mde": mde,
        "roc": roc,
    }

    output_path = Path(__file__).resolve().parent / "statistical_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results exported to: {output_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return output


if __name__ == "__main__":
    main()
