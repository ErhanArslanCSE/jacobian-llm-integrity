"""Perturbation types for weight-space anomaly injection.

All perturbations are applied IN-PLACE to avoid cloning the full model.
The pattern is: save original -> perturb -> measure -> restore.

Each perturbation satisfies ||ΔW||_F / ||W||_F = epsilon (relative norm).
"""

import torch
import torch.nn as nn


def _get_weight_params(layer):
    """Get all weight parameters from a transformer layer.

    Returns list of (name, param) tuples for parameters we want to perturb.
    We target the main linear layers: q_proj, k_proj, v_proj, o_proj,
    gate_proj, up_proj, down_proj.
    """
    targets = []
    for name, param in layer.named_parameters():
        if param.ndim == 2:  # Only weight matrices, not biases or norms
            targets.append((name, param))
    return targets


def save_weights(layer):
    """Save copies of all weight matrices in a layer."""
    saved = {}
    for name, param in _get_weight_params(layer):
        saved[name] = param.data.clone()
    return saved


def restore_weights(layer, saved):
    """Restore weight matrices from saved copies."""
    param_dict = dict(layer.named_parameters())
    for name, original in saved.items():
        param_dict[name].data.copy_(original)


def _apply_perturbation_to_param(param, delta):
    """Apply additive perturbation to a parameter in-place."""
    param.data.add_(delta)


def perturb_gaussian(layer, epsilon, seed):
    """Gaussian perturbation: ΔW ~ N(0, σ²I), scaled to ||ΔW||/||W|| = ε.

    This is the most generic perturbation — tests whether the Jacobian
    is sensitive to arbitrary weight changes.
    """
    rng = torch.Generator(device=layer.self_attn.q_proj.weight.device)
    rng.manual_seed(seed)

    saved = save_weights(layer)
    for name, param in _get_weight_params(layer):
        W = param.data
        # Use torch.randn with explicit shape/device/dtype (randn_like doesn't accept generator in all PyTorch versions)
        if W.device.type == 'cuda':
            delta = torch.randn(W.shape, device=W.device, dtype=W.dtype, generator=rng)
        else:
            torch.manual_seed(seed + hash(name) % (2**31))
            delta = torch.randn(W.shape, device=W.device, dtype=W.dtype)
        # Scale so ||delta||_F / ||W||_F = epsilon
        scale = epsilon * W.norm() / delta.norm()
        _apply_perturbation_to_param(param, delta * scale)

    return saved


def perturb_low_rank(layer, epsilon, rank, seed):
    """Low-rank perturbation: ΔW = UVᵀ where U,V have `rank` columns.

    Simulates a focused backdoor that modifies weights along a
    low-dimensional subspace.
    """
    device = layer.self_attn.q_proj.weight.device
    rng = torch.Generator(device='cpu')  # Generate on CPU for reproducibility
    rng.manual_seed(seed)

    saved = save_weights(layer)
    for name, param in _get_weight_params(layer):
        W = param.data
        m, n = W.shape
        # Generate U (m x rank) and V (n x rank) on CPU, then move
        U = torch.randn(m, rank, generator=rng, dtype=torch.float32)
        V = torch.randn(n, rank, generator=rng, dtype=torch.float32)
        delta = (U @ V.T).to(device=device, dtype=W.dtype)
        # Scale
        scale = epsilon * W.norm() / delta.norm()
        _apply_perturbation_to_param(param, delta * scale)

    return saved


def perturb_low_rank_1(layer, epsilon, seed):
    """Rank-1 perturbation."""
    return perturb_low_rank(layer, epsilon, rank=1, seed=seed)


def perturb_low_rank_4(layer, epsilon, seed):
    """Rank-4 perturbation."""
    return perturb_low_rank(layer, epsilon, rank=4, seed=seed)


def perturb_activation_matched(layer, epsilon, seed, model=None,
                                prompts=None, layer_idx=None, device=None):
    """Activation-matched perturbation via projected gradient descent.

    Simulates a sophisticated adversary (Bailey et al. 2024 style) who
    optimizes ΔW to minimize activation change while maintaining ||ΔW||/||W|| = ε.

    This is the critical NIAC scenario: weights are changed (computation differs)
    but activations look normal. If Jacobian monitoring works, SNR_J should
    remain detectable even when SNR_act is driven near zero.

    Method:
        1. Collect calibration data: layer inputs, kwargs, and target outputs
        2. Initialize ΔW ~ N(0,σ²I) scaled to ε (same as gaussian)
        3. Use Adam to minimize Σ_x ||layer(h_in; W+ΔW) - layer(h_in; W)||²
           via torch.func.functional_call (preserves gradient graph through ΔW)
        4. After each step, project ΔW back onto ||ΔW|| = ε·||W|| sphere
        5. Apply final optimized ΔW in-place
    """
    from torch.func import functional_call
    from .layer_wrapper import collect_hidden_states, collect_layer_outputs

    saved = save_weights(layer)

    if prompts is None or model is None:
        restore_weights(layer, saved)
        return perturb_gaussian(layer, epsilon, seed)

    _SKIP_KEYS = {"past_key_values", "use_cache", "cache_position"}
    n_cal = min(len(prompts), 50)  # Use up to 50 prompts for calibration
    n_steps = 1000
    lr = 1e-2

    # ── Step 1: Collect calibration data ──
    cal_data = []  # list of (h_input, clean_kwargs, h_target_output, last_token_idx)
    for prompt in prompts[:n_cal]:
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)

        hidden_dict, kwargs_dict = collect_hidden_states(
            model, input_ids, attention_mask, [layer_idx]
        )
        output_dict = collect_layer_outputs(
            model, input_ids, attention_mask, [layer_idx]
        )

        h_in = hidden_dict[layer_idx].detach()
        h_target = output_dict[layer_idx].detach()
        clean_kw = {k: v for k, v in kwargs_dict[layer_idx].items()
                    if k not in _SKIP_KEYS}
        last_tok = attention_mask.sum(dim=1).long().item() - 1

        cal_data.append((h_in, clean_kw, h_target, last_tok))

    if not cal_data:
        restore_weights(layer, saved)
        return perturb_gaussian(layer, epsilon, seed)

    # ── Step 2: Initialize deltas in float32 for precise optimization ──
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)

    deltas = {}
    target_norms = {}
    weight_names = []

    for name, param in _get_weight_params(layer):
        W = saved[name]
        d = torch.randn(W.shape, generator=rng, dtype=torch.float32, device='cpu')
        d = d.to(device=device)
        tn = epsilon * W.float().norm()
        d = d * (tn / d.norm())
        deltas[name] = nn.Parameter(d)
        target_norms[name] = tn
        weight_names.append(name)

    # ── Step 3: Projected gradient descent with cosine LR ──
    optimizer = torch.optim.Adam(deltas.values(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    for step in range(n_steps):
        total_loss = 0.0

        for h_in, clean_kw, h_target, last_tok in cal_data:
            # Build modified parameter dict: W_orig + delta for each weight
            modified = {}
            for name in weight_names:
                w_orig = saved[name]
                # Compute in float32, cast back to model dtype
                modified[name] = (w_orig.float() + deltas[name]).to(dtype=w_orig.dtype)

            # Run layer with modified weights (gradient flows through deltas)
            out_tuple = functional_call(layer, modified, (h_in,), kwargs=clean_kw)
            out = out_tuple[0]
            if out.dim() == 2 and h_in.dim() == 3:
                out = out.unsqueeze(0)

            # MSE loss at LAST TOKEN only (matches how activation monitoring measures)
            out_last = out[0, last_tok, :].float()
            tgt_last = h_target[0, last_tok, :].float()
            total_loss = total_loss + (out_last - tgt_last).pow(2).mean()

        total_loss = total_loss / len(cal_data)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Project each delta back onto ε-norm sphere
        with torch.no_grad():
            for name in weight_names:
                d = deltas[name]
                tn = target_norms[name]
                d.data.mul_(tn / d.data.norm().clamp(min=1e-12))

    # ── Step 4: Apply final optimized perturbation ──
    final_loss = total_loss.item()
    with torch.no_grad():
        for name, param in _get_weight_params(layer):
            delta_final = deltas[name].detach().to(dtype=param.dtype)
            param.data.copy_(saved[name] + delta_final)

    print(f"    activation_matched: {n_steps} steps, "
          f"final_loss={final_loss:.6f}, n_cal={len(cal_data)}")

    return saved


# Registry of perturbation functions
PERTURBATION_REGISTRY = {
    "gaussian": perturb_gaussian,
    "low_rank_1": perturb_low_rank_1,
    "low_rank_4": perturb_low_rank_4,
    "activation_matched": perturb_activation_matched,
}


def apply_perturbation(layer, ptype, epsilon, seed, **kwargs):
    """Apply a named perturbation type. Returns saved weights for restoration."""
    if ptype not in PERTURBATION_REGISTRY:
        raise ValueError(f"Unknown perturbation type: {ptype}. "
                         f"Available: {list(PERTURBATION_REGISTRY.keys())}")
    fn = PERTURBATION_REGISTRY[ptype]
    return fn(layer, epsilon, seed, **kwargs)
