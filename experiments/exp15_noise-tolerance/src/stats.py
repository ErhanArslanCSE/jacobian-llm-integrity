"""Decision test of Definition 1 of the paper (same implementation as exp9/src/metrics.py)."""

import numpy as np
from scipy import stats


def one_sample_ttest(values):
    """One-sided one-sample t-test of H0: mean = 0 against H1: mean > 0."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)

    if se < 1e-15:
        if mean > 0:
            return {"t_stat": float("inf"), "p_value": 0.0, "mean": mean, "std": 0.0}
        return {"t_stat": 0.0, "p_value": 1.0, "mean": mean, "std": 0.0}

    t_stat = mean / se
    p_value = float(stats.t.sf(t_stat, df=n - 1))
    return {"t_stat": float(t_stat), "p_value": p_value, "mean": mean, "std": std}


def monitor_rejects(values, alpha):
    """Definition 1: some statistic is positive and the t-test rejects at level alpha."""
    values = np.asarray(values, dtype=np.float64)
    return bool(np.any(values > 0) and one_sample_ttest(values)["p_value"] < alpha)
