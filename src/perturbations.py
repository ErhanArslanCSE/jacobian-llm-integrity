"""Perturbation types for weight-space anomaly injection.

All perturbations are applied IN-PLACE to avoid cloning the full model.
The pattern is: save original -> perturb -> measure -> restore.

Each perturbation satisfies ||DeltaW||_F / ||W||_F = epsilon (relative norm).
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
    """Gaussian perturbation: DeltaW ~ N(0, sigma^2 I), scaled to ||DeltaW||/||W|| = eps.

    This is the most generic perturbation -- tests whether the Jacobian
    is sensitive to arbitrary weight changes.
    """
    rng = torch.Generator(device=next(layer.parameters()).device)
    rng.manual_seed(seed)

    saved = save_weights(layer)
    for name, param in _get_weight_params(layer):
        W = param.data
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
    """Low-rank perturbation: DeltaW = UV^T where U,V have `rank` columns."""
    device = next(layer.parameters()).device
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)

    saved = save_weights(layer)
    for name, param in _get_weight_params(layer):
        W = param.data
        m, n = W.shape
        U = torch.randn(m, rank, generator=rng, dtype=torch.float32)
        V = torch.randn(n, rank, generator=rng, dtype=torch.float32)
        delta = (U @ V.T).to(device=device, dtype=W.dtype)
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
    optimizes DeltaW to minimize activation change while maintaining ||DeltaW||/||W|| = eps.
    """
    from torch.func import functional_call
    from .layer_wrapper import collect_hidden_states, collect_layer_outputs

    saved = save_weights(layer)

    if prompts is None or model is None:
        restore_weights(layer, saved)
        return perturb_gaussian(layer, epsilon, seed)

    _SKIP_KEYS = {"past_key_values", "use_cache", "cache_position"}
    n_cal = min(len(prompts), 50)
    n_steps = 1000
    lr = 1e-2

    # Step 1: Collect calibration data
    cal_data = []
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

    # Step 2: Initialize deltas in float32
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

    # Step 3: Projected gradient descent
    optimizer = torch.optim.Adam(deltas.values(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    for step in range(n_steps):
        total_loss = 0.0

        for h_in, clean_kw, h_target, last_tok in cal_data:
            modified = {}
            for name in weight_names:
                w_orig = saved[name]
                modified[name] = (w_orig.float() + deltas[name]).to(dtype=w_orig.dtype)

            out_tuple = functional_call(layer, modified, (h_in,), kwargs=clean_kw)
            out = out_tuple[0]
            if out.dim() == 2 and h_in.dim() == 3:
                out = out.unsqueeze(0)

            out_last = out[0, last_tok, :].float()
            tgt_last = h_target[0, last_tok, :].float()
            total_loss = total_loss + (out_last - tgt_last).pow(2).mean()

        total_loss = total_loss / len(cal_data)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for name in weight_names:
                d = deltas[name]
                tn = target_norms[name]
                d.data.mul_(tn / d.data.norm().clamp(min=1e-12))

    # Step 4: Apply final optimized perturbation
    final_loss = total_loss.item()
    with torch.no_grad():
        for name, param in _get_weight_params(layer):
            delta_final = deltas[name].detach().to(dtype=param.dtype)
            param.data.copy_(saved[name] + delta_final)

    print(f"    activation_matched: {n_steps} steps, "
          f"final_loss={final_loss:.6f}, n_cal={len(cal_data)}")

    return saved


# ── Single-sublayer Gaussian perturbation (for fair comparison) ──

def perturb_gaussian_sublayer(layer, epsilon, seed, sublayer_path="mlp.gate_proj"):
    """Gaussian perturbation applied to a SINGLE sublayer only.

    For fair comparison with null-space perturbation, which also targets
    one sublayer. Both use the same epsilon on the same weight matrix.

    Args:
        layer: Transformer decoder layer
        epsilon: Relative perturbation magnitude ||DeltaW||/||W||
        seed: Random seed
        sublayer_path: Which sublayer to perturb (default: mlp.gate_proj)

    Returns:
        saved: Dict of saved weights for restoration
    """
    saved = save_weights(layer)

    # Navigate to sublayer
    sublayer = layer
    for part in sublayer_path.split("."):
        sublayer = getattr(sublayer, part)

    W = sublayer.weight.data
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)

    delta = torch.randn(W.shape, generator=rng, dtype=torch.float32)
    delta = delta.to(device=W.device, dtype=W.dtype)
    scale = epsilon * W.norm() / delta.norm()
    sublayer.weight.data.add_(delta * scale)

    actual_ratio = (delta * scale).float().norm().item() / W.float().norm().item()
    print(f"  Gaussian sublayer: applied to {sublayer_path}, "
          f"||DeltaW||/||W||={actual_ratio:.6f} (target={epsilon})")

    return saved


# ── Null-space perturbation (NEW for exp2) ──

def collect_sublayer_inputs(model, prompts, layer_idx, sublayer_path, device):
    """Collect real inputs to a specific sublayer during forward pass.

    For example, mlp.gate_proj receives LN_2(residual), not the raw hidden states.
    This function uses hooks to capture exactly what each sublayer gets.

    Args:
        model: The transformer model
        prompts: List of prompt dicts (input_ids, attention_mask)
        layer_idx: Which transformer layer (e.g. 12)
        sublayer_path: Dot-separated path within the layer (e.g. "mlp.gate_proj")
        device: torch device

    Returns:
        H: Tensor of shape (n_prompts, d_in) — last-token inputs stacked
    """
    layer = model.model.layers[layer_idx]

    # Navigate to sublayer
    sublayer = layer
    for part in sublayer_path.split("."):
        sublayer = getattr(sublayer, part)

    inputs_list = []

    def hook_fn(module, args, kwargs=None):
        # args[0] is the input tensor: (batch, seq_len, d_in) or (batch*seq_len, d_in)
        inp = args[0].detach()
        if inp.dim() == 3:
            # Get last token input using attention_mask stored in closure
            inp_last = inp[0, hook_fn._last_token_idx, :].clone()
        elif inp.dim() == 2:
            inp_last = inp[hook_fn._last_token_idx, :].clone()
        else:
            inp_last = inp.flatten()[:sublayer.in_features].clone()
        inputs_list.append(inp_last.cpu().float())

    handle = sublayer.register_forward_pre_hook(hook_fn)

    for prompt in prompts:
        input_ids = prompt["input_ids"].to(device)
        attention_mask = prompt["attention_mask"].to(device)
        last_tok = attention_mask.sum(dim=1).long().item() - 1
        hook_fn._last_token_idx = last_tok

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

    handle.remove()

    H = torch.stack(inputs_list, dim=0)  # (n_prompts, d_in)
    return H


def perturb_null_space(layer, epsilon, seed, model=None, prompts=None,
                       layer_idx=None, sublayer_path="mlp.gate_proj", device=None):
    """Null-space perturbation: DeltaW lies in the null space of calibration inputs.

    This guarantees ZERO activation change on calibration data, while still
    modifying the weight matrix. The key question: does this change the Jacobian?

    Algorithm:
        1. Collect sublayer inputs H from calibration prompts: (n_cal, d_in)
        2. SVD of H: H = U S V^T
        3. Null space = columns of V corresponding to zero (or near-zero) singular values
           i.e., V[:, rank:] where rank = number of significant singular values
        4. Construct DeltaW with rows in the null space: DeltaW = A @ N^T
           where N = null space basis, A = random matrix
        5. Scale to ||DeltaW||/||W|| = epsilon
        6. Apply only to the target sublayer

    Math guarantee: For calibration input h_i, DeltaW @ h_i = 0
    because h_i is in the column space of H, and DeltaW's rows are in the null space.

    Args:
        layer: Transformer decoder layer
        epsilon: Relative perturbation magnitude
        seed: Random seed
        model: Model (needed for input collection)
        prompts: Calibration prompts
        layer_idx: Layer index
        sublayer_path: Which sublayer to perturb (default: mlp.gate_proj)
        device: torch device

    Returns:
        saved: Dict of saved weights for restoration
    """
    saved = save_weights(layer)

    if model is None or prompts is None:
        print("  WARNING: null_space requires model and prompts, falling back to gaussian")
        restore_weights(layer, saved)
        return perturb_gaussian(layer, epsilon, seed)

    # Step 1: Collect sublayer inputs
    H = collect_sublayer_inputs(model, prompts, layer_idx, sublayer_path, device)
    n_cal, d_in = H.shape
    print(f"  Null-space: collected {n_cal} calibration inputs, d_in={d_in}")

    # Step 2: SVD of H^T (so we get V for H = U S V^T)
    # H is (n_cal, d_in). We want the right singular vectors of H.
    # torch.linalg.svd(H) gives U(n_cal,n_cal), S(min), V^T(d_in, d_in)
    # The null space of H (as a linear map from R^d_in -> R^n_cal) is
    # the columns of V corresponding to zero singular values.
    _, S, Vt = torch.linalg.svd(H.float(), full_matrices=True)
    V = Vt.T  # (d_in, d_in) — columns are right singular vectors

    # Determine rank using threshold
    # Singular values below this are considered zero
    tol = 1e-5 * S[0].item()  # Relative to largest singular value
    rank = (S > tol).sum().item()
    null_dim = d_in - rank
    print(f"  Null-space: H rank={rank}, null_dim={null_dim}, "
          f"sigma_max={S[0].item():.4f}, sigma_min_nonzero={S[rank-1].item():.6f}")

    if null_dim == 0:
        print("  WARNING: null space is empty (H is full rank). Falling back to gaussian.")
        restore_weights(layer, saved)
        return perturb_gaussian(layer, epsilon, seed)

    # Step 3: Null space basis — columns of V corresponding to zero singular values
    N = V[:, rank:]  # (d_in, null_dim)

    # Step 4: Navigate to the target sublayer's weight
    sublayer = layer
    sublayer_name = None
    for part in sublayer_path.split("."):
        sublayer = getattr(sublayer, part)

    # Find the full parameter name
    for name, param in layer.named_parameters():
        if sublayer_path.replace(".", ".") in name and "weight" in name:
            sublayer_name = name
            break

    if sublayer_name is None:
        # Fallback: construct name
        sublayer_name = sublayer_path + ".weight"

    W = sublayer.weight.data  # (d_out, d_in)
    d_out = W.shape[0]

    # Step 5: Construct DeltaW = A @ N^T where A is random (d_out, null_dim)
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    A = torch.randn(d_out, null_dim, generator=rng, dtype=torch.float32)
    N_cpu = N.cpu().float()
    DeltaW = A @ N_cpu.T  # (d_out, d_in)

    # Step 6: Scale to ||DeltaW||/||W|| = epsilon
    W_norm = W.float().cpu().norm().item()
    DeltaW_norm = DeltaW.norm().item()
    scale = epsilon * W_norm / (DeltaW_norm + 1e-12)
    DeltaW = DeltaW * scale

    # Apply to weight
    DeltaW_device = DeltaW.to(device=W.device, dtype=W.dtype)
    sublayer.weight.data.add_(DeltaW_device)

    actual_ratio = DeltaW_device.float().norm().item() / (W_norm + 1e-12)
    print(f"  Null-space: applied to {sublayer_path}, "
          f"||DeltaW||/||W||={actual_ratio:.6f} (target={epsilon})")

    # Step 7: Verify on calibration data
    # DeltaW @ h_i should be ~0 for all calibration inputs
    H_device = H.to(device=W.device, dtype=torch.float32)
    DeltaW_f = DeltaW_device.float()
    residuals = DeltaW_f @ H_device.T  # (d_out, n_cal)
    max_residual = residuals.abs().max().item()
    mean_residual = residuals.abs().mean().item()
    print(f"  Null-space verification: max|DeltaW @ h_cal|={max_residual:.2e}, "
          f"mean={mean_residual:.2e}")

    return saved


# Registry of perturbation functions
PERTURBATION_REGISTRY = {
    "gaussian": perturb_gaussian,
    "gaussian_sublayer": perturb_gaussian_sublayer,
    "low_rank_1": perturb_low_rank_1,
    "low_rank_4": perturb_low_rank_4,
    "activation_matched": perturb_activation_matched,
    "null_space": perturb_null_space,
}


def apply_perturbation(layer, ptype, epsilon, seed, **kwargs):
    """Apply a named perturbation type. Returns saved weights for restoration."""
    if ptype not in PERTURBATION_REGISTRY:
        raise ValueError(f"Unknown perturbation type: {ptype}. "
                         f"Available: {list(PERTURBATION_REGISTRY.keys())}")
    fn = PERTURBATION_REGISTRY[ptype]
    return fn(layer, epsilon, seed, **kwargs)
