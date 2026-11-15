"""Activation statistics at every decoder layer and their per-layer decisions (as in exp12).

Statistics per prompt and decoder layer, relative to the approved model:
  last      ||a_T - r_T|| / ||r_T||            at the last token T (s_act of the paper)
  pos_mean  mean_t ||a_t - r_t|| / ||r_t||     over all positions
  pos_max   max_t  ||a_t - r_t|| / ||r_t||     over all positions
  kl_last   KL(p_ref || p) of the next-token distribution at the last token
"""

import numpy as np
import torch

from .stats import monitor_rejects, one_sample_ttest

STATS = ("last", "pos_mean", "pos_max")


def run_with_layer_outputs(model, prompt, device):
    """One forward pass; outputs of every decoder layer (L, T, d) and last-token log-probs."""
    layers = model.model.layers
    outputs = [None] * len(layers)
    handles = []

    def make_hook(idx):
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            outputs[idx] = (out[0] if out.dim() == 3 else out).detach()
        return hook

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(idx)))
    try:
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        for h in handles:
            h.remove()

    last = attention_mask.sum(dim=1).long().item() - 1
    return torch.stack(outputs, dim=0), torch.log_softmax(logits[0, last].float(), dim=-1)


def measure_activations(model, prompts, refs, ref_logps, device):
    """Statistics of every prompt against the references: stat -> (n_prompts, L), kl_last -> (n,)."""
    per_stat = {s: [] for s in STATS}
    kls = []
    for prompt, ref, ref_logp in zip(prompts, refs, ref_logps):
        out, logp = run_with_layer_outputs(model, prompt, device)
        rel = (out.float() - ref.float()).norm(dim=-1) / ref.float().norm(dim=-1).clamp_min(1e-10)
        per_stat["last"].append(rel[:, -1].cpu().numpy())
        per_stat["pos_mean"].append(rel.mean(dim=1).cpu().numpy())
        per_stat["pos_max"].append(rel.max(dim=1).values.cpu().numpy())
        kls.append(float((ref_logp.exp() * (ref_logp - logp)).sum().item()))
    result = {s: np.stack(v, axis=0) for s, v in per_stat.items()}
    result["kl_last"] = np.array(kls)
    return result


def layer_table(ns_vals, g_vals, alpha, threshold):
    """Per-layer decision quantities; ns_vals, g_vals: (n_prompts, L) for an edit and its control."""
    rows = []
    for l in range(ns_vals.shape[1]):
        ns_t = one_sample_ttest(ns_vals[:, l])
        g_mean = float(np.mean(g_vals[:, l]))
        suppression = 1.0 - ns_t["mean"] / g_mean if g_mean > 0 else None
        rows.append({
            "layer": l,
            "ns_mean": ns_t["mean"],
            "ns_std": ns_t["std"],
            "ns_p": ns_t["p_value"],
            "ns_rejects": monitor_rejects(ns_vals[:, l], alpha),
            "g_mean": g_mean,
            "suppression": suppression,
            "evaded": suppression is not None and suppression > threshold,
        })
    return rows


def summarize_layers(rows, edited_layer):
    """Every-layer audit (assumption 5) over the layers the edit can reach (>= edited layer)."""
    scope = [r for r in rows if r["layer"] >= edited_layer]
    not_evaded = [r["layer"] for r in scope if not r["evaded"]]
    return {
        "edited_layer_suppression": scope[0]["suppression"],
        "next_layer_suppression": scope[1]["suppression"] if len(scope) > 1 else None,
        "min_suppression_downstream": min(r["suppression"] for r in scope),
        "first_layer_not_evaded": not_evaded[0] if not_evaded else None,
        "every_layer_audit_evaded": not not_evaded,
        "max_ns_mean_downstream": max(r["ns_mean"] for r in scope),
    }
