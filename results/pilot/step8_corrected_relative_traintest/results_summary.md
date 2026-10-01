# Step 8: Corrected Formulas + Train/Test Split — 2026-02-26

## What Changed (Fixes from Critical Self-Audit)

### Fix 1: Relative Normalization
- **Before:** Absolute metrics — `act_diff = ||h_anom - h_norm||`, `jvp_diff = mean(|jvp_anom - jvp_norm|)`
- **After:** Relative (dimensionless) metrics matching research plan formulas:
  - `act_diff = ||h_anom - h_norm|| / ||h_norm||` (fractional activation change)
  - `jvp_diff = mean(|jvp_anom - jvp_norm|) / mean(jvp_norm)` (fractional JVP change)
- **Why:** Both metrics are now dimensionless, comparable across layers/models, and match `02_research_plan.md`

### Fix 2: Train/Test Split
- **Before:** Same 50 prompts used for BOTH adversary GD calibration AND evaluation
- **After:** 60 prompts split into 30 train (adversary calibration) + 30 test (evaluation)
- **Why:** Using same data for calibration and evaluation is data leakage — artificially inflates adversary success

### Fix 3: Honest Metrics
- **Before:** "768x Jacobian advantage" compared raw means across different perturbation types — misleading
- **After:** Only report SNR ratio = SNR_J / SNR_act, the metric defined in research plan

## Setup
- Model: meta-llama/Llama-3.2-1B, bfloat16
- Hardware: RunPod H200 SXM
- Config: configs/corrected_v1.yaml
- Prompts: 60 total (30 train + 30 test), WikiText-103
- Directions: 30, Layers: [4, 8, 12], Target: layer 12
- GD: 1000 steps, Adam lr=1e-2, cosine annealing (min=1e-4)
- Runtime: 307.6s (5.1 min)

## Full Results (Layer 12 — only layer affected by perturbation)

### Gaussian (baseline — random perturbation)
| ε    | act_mean | act_std | SNR_act | jvp_mean | jvp_std | SNR_J | Ratio | Decision |
|------|----------|---------|---------|----------|---------|-------|-------|----------|
| 0.01 | 0.00976  | 0.00160 | 6.100   | 0.000396 | 0.000307| 1.287 | 0.211 | NO-GO    |
| 0.05 | 0.04240  | 0.00698 | 6.075   | 0.001789 | 0.000608| 2.941 | 0.484 | NO-GO    |
| 0.1  | 0.08497  | 0.01413 | 6.015   | 0.005615 | 0.001765| 3.182 | 0.529 | NO-GO    |

### Activation-matched (1000-step GD, trained on 30 prompts, tested on 30 different)
| ε    | act_mean | act_std | SNR_act | jvp_mean | jvp_std | SNR_J | Ratio | Decision |
|------|----------|---------|---------|----------|---------|-------|-------|----------|
| 0.01 | 0.06159  | 0.01534 | 4.015   | 0.01531  | 0.01593 | 0.961 | 0.239 | NO-GO    |
| 0.05 | 0.18547  | 0.04293 | 4.320   | 0.05044  | 0.03223 | 1.565 | 0.362 | NO-GO    |
| 0.1  | 0.25782  | 0.04711 | 5.473   | 0.04492  | 0.02497 | 1.799 | 0.329 | NO-GO    |

### Layers 4 and 8
All zeros — expected. Perturbation is applied to layer 12 only. Layers 4 and 8 outputs
are computed BEFORE the perturbed layer, so they are unchanged.

## Overall Verdict: NO-GO (0/18 conditions met GO threshold)

## Key Observations

### 1. Activation-matched does NOT evade activation monitoring (with relative metrics)
With absolute metrics (Step 7), we saw act_mean drop from 0.479 to 0.020 — a 97% reduction.
With relative metrics, we see:
- gaussian ε=0.05: act_mean = 0.042 (4.2% fractional change)
- actmatch ε=0.05: act_mean = 0.185 (18.5% fractional change)

**Activation-matched actually has HIGHER relative activation change than gaussian!**
This is the opposite of what we thought. The GD optimizer minimizes absolute L2 distance,
not relative change. Since the adversary's ΔW norm is 5% of ||W||, a 4.2% relative
activation change (gaussian) is already small. The GD optimization isn't reducing it
further — it's actually increasing it.

### 2. SNR ratio is consistently < 1
- Gaussian: ratio ∈ [0.21, 0.53]
- Activation-matched: ratio ∈ [0.24, 0.36]
- GO threshold requires ratio > 3

**Activations are ALWAYS more sensitive (higher SNR) than Jacobians** for detecting
weight perturbations, regardless of perturbation type.

### 3. Why the ratio is < 1 (fundamental reason)
- SNR_act ≈ 4-6 across all conditions: deterministic ΔW produces consistent activation
  changes (low variance across prompts → high SNR)
- SNR_J ≈ 1-3: JVP changes have higher variance relative to mean (random directions
  introduce variability)
- The JVP metric averages over k=30 random directions, which injects sampling noise
  into the signal, reducing SNR

### 4. Comparison: Step 7 absolute vs Step 8 relative
| Condition | Step 7 (abs) act_mean | Step 8 (rel) act_mean | Step 7 Ratio | Step 8 Ratio |
|-----------|----------------------|----------------------|--------------|--------------|
| gauss 0.05 | 0.479               | 0.042                | 0.251        | 0.484        |
| actmatch 0.05 | 0.020            | 0.185                | 0.506        | 0.362        |

The relative normalization actually:
- Reduces gaussian act_mean (activation norms are large, ~11x scaling)
- Increases actmatch act_mean (GD was minimizing absolute, not relative)
- Changes ratios but direction of NO-GO is unchanged

### 5. Train/test split effect
With only 30 calibration prompts (vs 50 before), GD convergence is comparable
(final_loss = 0.000000). But the test set is now unseen by the adversary — a more
realistic evaluation. The NO-GO verdict holds under this more rigorous setup.

## Honest Assessment

The SNR ratio metric, as defined in the research plan, produces a clear **NO-GO**:
- 8 experiment steps over 7 hours of iterative development
- Fixed 5 bugs (INPUT→OUTPUT, SVD→GD, all-position→last-token, absolute→relative, data leak)
- Tried TinyLlama and Llama-3.2-1B
- Tried ε from 0.001 to 0.2
- Tried 300 to 1000 GD steps with cosine LR
- **NEVER achieved SNR_J / SNR_act > 1, let alone > 3**

The Jacobian is NOT more sensitive than activations for detecting weight perturbations
in the SNR framework. The random-direction JVP approach introduces too much variance
compared to the deterministic activation signal.

## Possible Directions (if pursuing despite NO-GO)
1. **Change the detection paradigm:** Instead of SNR ratio, use a binary classifier
   (e.g., threshold on activation + JVP features jointly)
2. **Spectral analysis of JVP:** Instead of comparing JVP norms, analyze the full
   distribution or spectral signature of JVPs
3. **Different Jacobian metric:** Frobenius norm of full Jacobian instead of random JVPs
4. **Multi-layer propagation:** Measure how perturbation propagates through subsequent layers
5. **Accept the result:** The NIAC hypothesis may be wrong — activation monitoring may
   be sufficient for this threat model
