"""Edits of one sublayer: Gaussian control and L3 null-space edit (identical to exp9), and the
all-position adaptive adversary of exp13.

perturb_gaussian_sublayer and perturb_null_space draw their random matrices exactly as
exp9/src/perturbations.py does (CPU generator, same seed, same call order), so with the
same model, prompts and seed they reproduce the edits behind Tables II and III of the paper.
All edits are applied in place: save -> perturb -> measure -> restore.
"""

import torch


def get_sublayer(layer, sublayer_path):
    """Navigate from a decoder layer to a sublayer, e.g. 'mlp.gate_proj'."""
    sublayer = layer
    for part in sublayer_path.split("."):
        sublayer = getattr(sublayer, part)
    return sublayer


def save_weights(layer):
    """Save copies of all weight matrices of a decoder layer."""
    return {name: p.data.clone() for name, p in layer.named_parameters() if p.ndim == 2}


def restore_weights(layer, saved):
    """Restore weight matrices from saved copies."""
    params = dict(layer.named_parameters())
    for name, original in saved.items():
        params[name].data.copy_(original)


def weights_equal(layer, saved):
    """True if every saved weight matrix is bitwise equal to the current one."""
    params = dict(layer.named_parameters())
    return all(torch.equal(params[name].data, original) for name, original in saved.items())


def perturb_gaussian_sublayer(layer, epsilon, seed, sublayer_path):
    """Gaussian control: DeltaW ~ N(0, I) on one sublayer, scaled to ||DeltaW||/||W|| = eps."""
    sublayer = get_sublayer(layer, sublayer_path)
    W = sublayer.weight.data
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    delta = torch.randn(W.shape, generator=rng, dtype=torch.float32)
    delta = delta.to(device=W.device, dtype=W.dtype)
    scale = epsilon * W.norm() / delta.norm()
    sublayer.weight.data.add_(delta * scale)


def _sublayer_input_hook(sublayer, positions):
    """Register a pre-hook that stores the sublayer input at the given positions.

    positions: "last" (last token, as the paper's calibration inputs) or "all".
    Returns (handle, store, set_last) where set_last(idx) sets the last-token index.
    """
    store = []
    state = {"last": None}

    def hook(module, args):
        inp = args[0].detach()
        if inp.dim() == 3:
            inp = inp[0]
        if positions == "last":
            store.append(inp[state["last"], :].float().cpu().unsqueeze(0))
        else:
            store.append(inp.float().cpu())

    handle = sublayer.register_forward_pre_hook(hook)

    def set_last(idx):
        state["last"] = idx

    return handle, store, set_last


def collect_sublayer_inputs(model, prompts, layer_idx, sublayer_path, device, positions="last"):
    """Inputs of a sublayer on the prompts, as rows of a matrix.

    positions="last" returns (n_prompts, d_in): the calibration inputs h_i of Theorem 1.
    positions="all" returns (sum of prompt lengths, d_in): the inputs at every position.
    """
    sublayer = get_sublayer(model.model.layers[layer_idx], sublayer_path)
    handle, store, set_last = _sublayer_input_hook(sublayer, positions)
    try:
        for prompt in prompts:
            input_ids = prompt["input_ids"].to(device)
            attention_mask = prompt["attention_mask"].to(device)
            set_last(attention_mask.sum(dim=1).long().item() - 1)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return torch.cat(store, dim=0)


def perturb_null_space(layer, epsilon, seed, model, prompts, layer_idx, sublayer_path, device):
    """L3 edit: Gaussian matrix with rows projected onto the null space of the
    calibration inputs, scaled to ||DeltaW||/||W|| = eps (Section VI-A-2 of the paper).

    Returns the intended edit DeltaW (float32, CPU), before it is added to the weights in
    the model dtype, so that the caller can measure the rounding of W + DeltaW.
    """
    H = collect_sublayer_inputs(model, prompts, layer_idx, sublayer_path, device)
    d_in = H.shape[1]

    _, S, Vt = torch.linalg.svd(H.float(), full_matrices=True)
    V = Vt.T
    tol = 1e-5 * S[0].item()
    rank = (S > tol).sum().item()
    null_dim = d_in - rank
    if null_dim == 0:
        raise RuntimeError(f"Null space of the calibration inputs at layer {layer_idx} is empty")
    N = V[:, rank:]

    sublayer = get_sublayer(layer, sublayer_path)
    W = sublayer.weight.data
    d_out = W.shape[0]

    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    A = torch.randn(d_out, null_dim, generator=rng, dtype=torch.float32)
    DeltaW = A @ N.cpu().float().T

    W_norm = W.float().cpu().norm().item()
    DeltaW = DeltaW * (epsilon * W_norm / (DeltaW.norm().item() + 1e-12))

    sublayer.weight.data.add_(DeltaW.to(device=W.device, dtype=W.dtype))
    print(f"    null space: rank(H)={rank}, null_dim={null_dim}")
    return DeltaW


def last_token_null_basis(H_last):
    """Orthonormal basis (d_in, d_in - rank) of the null space of the last-token inputs,
    with the rank threshold of perturb_null_space."""
    _, S, Vt = torch.linalg.svd(H_last.float(), full_matrices=True)
    rank = (S > 1e-5 * S[0].item()).sum().item()
    return Vt.T[:, rank:]


def allpos_null_basis(H_last, H_all, m):
    """Adaptive adversary of exp13: m orthonormal input directions (d_in, m) that are exactly
    orthogonal to every last-token calibration input (so Theorem 1 still holds at the edited
    layer) and, within that null space, carry the least energy of the calibration inputs at
    all positions (so the edit changes the earlier positions, and through attention the later
    layers, as little as possible).

    Also returns the leakage ratio ||H_all B||_F^2 / (m ||H_all||_F^2 / d_in): the all-position
    energy that a Gaussian edit in span(B) sees, relative to an unconstrained Gaussian edit of
    the same norm (1 = no protection, 0 = exact all-position null space).
    """
    N = last_token_null_basis(H_last)
    G = H_all.float() @ N
    _, _, Vgt = torch.linalg.svd(G, full_matrices=True)
    B = N @ Vgt[-m:].T
    d_in = H_all.shape[1]
    leakage = (H_all.float() @ B).pow(2).sum() / (m * H_all.float().pow(2).sum() / d_in)
    return B, float(leakage)


def perturb_in_subspace(layer, epsilon, seed, B, sublayer_path):
    """Edit DeltaW = A B^T with A standard Gaussian (CPU generator, seed), so every row of
    DeltaW lies in span(B); scaled to ||DeltaW||/||W|| = eps. Returns the intended DeltaW."""
    sublayer = get_sublayer(layer, sublayer_path)
    W = sublayer.weight.data
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    A = torch.randn(W.shape[0], B.shape[1], generator=rng, dtype=torch.float32)
    DeltaW = A @ B.cpu().float().T
    DeltaW = DeltaW * (epsilon * W.float().cpu().norm().item() / (DeltaW.norm().item() + 1e-12))
    sublayer.weight.data.add_(DeltaW.to(device=W.device, dtype=W.dtype))
    return DeltaW
