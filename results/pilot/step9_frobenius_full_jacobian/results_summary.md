# Step 9: Full Jacobian Frobenius Norm — Last Chance Experiment — 2026-02-26

## Motivation
Steps 1-8 used k=30 random JVP projections — a scalar summary of the Jacobian.
The paper claims to use the "full Jacobian distribution," but the experiment
used the same scalar-summary approach we criticize others for (Park et al., GradNorm).

This step computes the **FULL d×d Jacobian matrix** using all d=2048 standard
basis JVP directions, then measures:

    jac_diff = ||J_anom - J_norm||_F / ||J_norm||_F

This is the research plan's exact formula — no approximations.

## Setup
- Model: meta-llama/Llama-3.2-1B, bfloat16, hidden_dim=2048
- Hardware: RunPod H200 SXM
- Config: configs/step9_frobenius.yaml
- Prompts: 60 (30 train + 30 test), WikiText-103
- Target layer: 12 (only layer perturbed)
- Full Jacobian: 2048 JVPs per prompt (standard basis vectors)
- Also computed: random JVP (k=30) for direct comparison
- GD: 1000 steps, Adam lr=1e-2, cosine annealing
- Runtime: 2425.9s (40.4 min) — 10.25s per prompt for full Jacobian

## Full Results (Layer 12)

### Gaussian (random perturbation)
| ε    | act_mean | SNR_act | jac_frob_mean | SNR_J_frob | **ratio_frob** | decision_frob | SNR_J_rand | ratio_rand |
|------|----------|---------|---------------|------------|----------------|---------------|------------|------------|
| 0.01 | 0.010    | 6.100   | 0.013         | 9.521      | **1.561**      | MARGINAL      | 1.287      | 0.211      |
| 0.05 | 0.042    | 6.075   | 0.060         | 9.664      | **1.591**      | MARGINAL      | 2.941      | 0.484      |
| 0.1  | 0.085    | 6.015   | 0.120         | 9.678      | **1.609**      | MARGINAL      | 3.182      | 0.529      |

### Activation-matched (1000-step GD, trained on 30, tested on 30 different)
| ε    | act_mean | SNR_act | jac_frob_mean | SNR_J_frob | **ratio_frob** | decision_frob | SNR_J_rand | ratio_rand |
|------|----------|---------|---------------|------------|----------------|---------------|------------|------------|
| 0.01 | 0.062    | 4.015   | 0.116         | 1.962      | **0.489**      | NO-GO         | 0.961      | 0.239      |
| 0.05 | 0.185    | 4.320   | 0.292         | 3.414      | **0.790**      | NO-GO         | 1.565      | 0.362      |
| 0.1  | 0.258    | 5.473   | 0.376         | 3.288      | **0.601**      | NO-GO         | 1.799      | 0.329      |

### Frobenius vs Random JVP Improvement
| Condition                | ratio_frob | ratio_random | improvement |
|--------------------------|------------|-------------|-------------|
| gaussian_eps0.01         | 1.561      | 0.211       | **7.4x**    |
| gaussian_eps0.05         | 1.591      | 0.484       | **3.3x**    |
| gaussian_eps0.1          | 1.609      | 0.529       | **3.0x**    |
| activation_matched_eps0.01| 0.489     | 0.239       | 2.0x        |
| activation_matched_eps0.05| 0.790     | 0.362       | 2.2x        |
| activation_matched_eps0.1 | 0.601     | 0.329       | 1.8x        |

## Overall: 0 GO, 3 MARGINAL, 3 NO-GO → Verdict: NO-GO

## Key Findings

### 1. Full Jacobian IS significantly better than random JVPs
The Frobenius norm improves ratios by **3-7x** for gaussian and **2x** for activation_matched.
All 3 gaussian conditions moved from NO-GO → MARGINAL. This validates our suspicion that
random JVP projections (k=30 out of d=2048) were undersampling the Jacobian signal.

### 2. For random perturbations, Jacobian achieves MARGINAL
SNR_J_frob (~9.5) consistently exceeds SNR_act (~6.1) for gaussian perturbations.
The full Jacobian changes more consistently across prompts than activations do.
ratio_frob ≈ 1.6 — well above 1 but below the GO threshold of 3.

### 3. For adversarial perturbations, Jacobian is still less sensitive
The GD optimization that minimizes activation change **also reduces Jacobian change**.
SNR_J_frob drops from ~9.6 (gaussian) to ~2-3.4 (actmatch) — a 3-5x reduction.
Meanwhile SNR_act drops from ~6.1 to ~4-5.5 — a smaller 1.1-1.5x reduction.
The adversary affects the Jacobian MORE than the activations.

### 4. Absolute Jacobian change IS larger than activation change
| Condition          | act_mean | jac_frob_mean | jac_frob / act |
|--------------------|----------|---------------|----------------|
| gaussian_eps0.05   | 0.042    | 0.060         | 1.4x           |
| actmatch_eps0.05   | 0.185    | 0.292         | 1.6x           |

The Jacobian changes MORE than activations in absolute terms. The lower SNR ratio
is because the Jacobian has higher variance across prompts (std/mean is larger).

### 5. The variance problem
| Condition          | act: std/mean | jac: std/mean |
|--------------------|---------------|---------------|
| gaussian_eps0.05   | 0.165         | 0.103         |
| actmatch_eps0.05   | 0.231         | 0.293         |

For gaussian: jac variance is LOWER → higher SNR → MARGINAL.
For actmatch: jac variance is HIGHER → lower SNR → NO-GO.
The adversary's GD introduces inconsistency in the Jacobian response.

## Honest Assessment

**The full Jacobian Frobenius norm is a MUCH better metric than random JVP projections.**
Steps 1-8's NO-GO was partially an artifact of the poor metric (k=30 random directions
sampling 1.5% of the Jacobian space).

**But the final verdict is still NO-GO** for the NIAC hypothesis:
- For random perturbations: Jacobian achieves MARGINAL (ratio ~1.6, need >3 for GO)
- For adversarial perturbations: Jacobian is less sensitive (ratio ~0.5-0.8)
- The adversary's activation evasion also reduces the Jacobian signal

**The NIAC hypothesis ("Jacobian detects what activations miss") is NOT supported
by these experiments.** When the adversary hides activation changes, the Jacobian
signal also degrades — they are NOT independent monitoring channels.

## What This Means for the Paper

1. **The completeness theorem (Theorem 2) is likely CORRECT** — the full Jacobian
   mathematically determines the computation. But the SNR framework doesn't capture this.
2. **The detection PROBLEM is not about sensitivity (SNR)** but about the detection
   PARADIGM. SNR ratio asks "which signal has higher consistency?" A better question
   might be "can a classifier using BOTH signals outperform either alone?"
3. **Possible pivot:** Instead of claiming "Jacobians are more sensitive than activations,"
   claim "Jacobians provide COMPLEMENTARY information" — and demonstrate via a multivariate
   detector (e.g., logistic regression on act_diff + jac_frob_diff features).
4. **The absolute Jacobian change is substantial (1.4-1.6x larger than activation change).**
   The issue is variance, not signal strength. A more sophisticated statistical test
   (multivariate, or Fisher's combined test) might yield a positive result.

## Comparison Across All 9 Steps
| Step | Metric | Best Ratio | Verdict |
|------|--------|-----------|---------|
| 1-3  | Absolute, scalar | 0.1-0.2 | NO-GO |
| 4-6  | Absolute, random JVP | 0.5-0.8 | NO-GO |
| 7    | Absolute, random JVP | 0.8 | NO-GO |
| 8    | Relative, random JVP | 0.5 | NO-GO |
| 9    | Relative, full Frobenius | **1.6** (gaussian) | **MARGINAL** |
| 9    | Relative, full Frobenius | 0.8 (actmatch) | NO-GO |
