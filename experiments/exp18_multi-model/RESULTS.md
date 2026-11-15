# Exp18 Results (against PREREGISTRATION.md)

18 runs: 3 models x 3 variants x 2 learning rates x seed 42; Llama-3.1-8B and Qwen2.5-7B on H200,
Mistral-7B-v0.3 on B200. Raw runs `results/<model>/runs/`, logs `results/exp18_output_*.log`,
tables: `python analyze_exp18.py results/<model>`. Clean perplexity (held-out): Llama-3.1-8B
30.77, Qwen2.5-7B 24.49, Mistral-7B-v0.3 40.43.

## Summary per model

| | Llama-3.1-8B | Qwen2.5-7B | Mistral-7B-v0.3 |
|---|---|---|---|
| L2 functional / every-layer audit evaded | 2/2 / 0/2 | 2/2 / 0/2 | 2/2 / 0/2 |
| AP1024 functional / evaded (cal) | 2/2 / 2/2 | 2/2 / 1/2 (lr 1e-3) | 2/2 / 2/2 |
| AP1024 evaded on held-out prompts | 0/2 | 0/2 | 0/2 |
| AP1024_pen1e2 functional / evaded | 2/2 / 1/2 | 2/2 / 0/2 | 2/2 / 1/2 |
| Tap test rejects (every trained layer) | 6/6 | 6/6 | 6/6 |
| Tap signal kept, AP1024 -> pen1e2 (selected) | 74.5% -> 39.8% | 58.1% -> none eligible | 61.8% -> 34.2% |

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| M1 | L2 functional, not evaded on all three | **Held** (worst-layer suppression -34% to -539%). |
| M2 | AP1024 functional at some lr on all three | **Held** (100% trigger in all 6 runs). |
| M3 | AP1024 evades on at least two of three | **Held on all three** (Llama 52-85%, Qwen 73% at lr 1e-3, Mistral 79-95%). |
| M4 | Where M3 fails, it fails at the smaller lr first | **Held** (Qwen: 22% at 2e-4, 73% at 1e-3). |
| M5 | Tap test detects every run | **Held** (18/18, every trained layer). |
| M6 | pen1e2 keeps less tap signal than AP1024 on every model | **Held** (all lr, all models). |

## Interpretation (pre-registered rule "M1, M3 and M5 hold on all three models")

The three findings of exp12-exp17 hold across three families and 1B-8B: the paper's backdoor is
seen by an every-layer activation audit; an adversary that also hides at the earlier positions
keeps a functional backdoor hidden from that audit on the calibration prompts; the tap test
detects every variant on a deterministic stack. As on Llama-3.2-1B, held-out prompts see every
AP backdoor, and the Jacobian-aware penalty trades activation evasion for a lower tap signal
(on Qwen no penalised run stays activation-evaded).

## Caveats

- One seed. Mistral's relative edit sizes are large (AP1024 eps 0.40 and 1.58; L2 up to 2.95),
  i.e. comparable to or larger than the gate_proj weights; the backdoor stays functional and
  perplexity drops (36.9, 33.7 vs. 40.4 clean) except L2 at lr 1e-3 (123.2).
- Each run uses clean references of its own machine (zero noise); the cross-architecture margin
  of the Jacobian-aware runs on these models was not measured (exp16 does this on Llama-3.2-1B).
