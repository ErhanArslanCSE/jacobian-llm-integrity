"""Sublayer inputs, the subspaces of the adversaries, and the Gaussian control (from exp13)."""

import torch


def get_sublayer(layer, sublayer_path):
    """Navigate from a decoder layer to a sublayer, e.g. 'mlp.gate_proj'."""
    sublayer = layer
    for part in sublayer_path.split("."):
        sublayer = getattr(sublayer, part)
    return sublayer


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
