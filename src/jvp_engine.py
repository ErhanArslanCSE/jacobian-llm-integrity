"""Core JVP computation via torch.func.

Computes Jacobian-vector products for transformer layers using
forward-mode automatic differentiation. This is the heart of the
Jacobian SNR experiment.

Supports two modes:
  - Sequential (chunk_size=None): 1 JVP per forward pass, d total. Safe but slow.
  - Batched (chunk_size>0): vmap over JVP directions for GPU parallelism.
    ~3-5x speedup on H200. Mathematically identical results.
"""

import torch
from torch.func import jvp

from .layer_wrapper import make_layer_fn


def compute_jvp_norms(layer, hidden_states, layer_kwargs,
                      num_directions, last_token_idx, seed, device):
    """Compute JVP norms for a single layer across multiple random directions.

    Uses last-token JVP: tangent vector is zero everywhere except at the
    last non-pad token position. This focuses sensitivity measurement on
    the token that matters for next-token prediction.

    Args:
        layer: Transformer decoder layer module
        hidden_states: Input hidden states (1, seq_len, hidden_dim)
        layer_kwargs: Dict of kwargs captured from model's forward pass
                      (attention_mask, position_ids, position_embeddings, etc.)
        num_directions: Number of random tangent directions (k)
        last_token_idx: Index of last non-pad token
        seed: Random seed for reproducible directions
        device: torch device

    Returns:
        Tensor of shape (num_directions,) containing ||J @ v||_2 for each direction
    """
    layer_fn = make_layer_fn(layer, layer_kwargs)

    batch_size, seq_len, hidden_dim = hidden_states.shape
    dtype = hidden_states.dtype

    jvp_norms = []
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    for i in range(num_directions):
        # Create tangent: zero everywhere except at last token position
        tangent = torch.zeros_like(hidden_states)
        v = torch.randn(hidden_dim, generator=rng, device=device, dtype=dtype)
        v = v / v.norm()  # Unit vector
        tangent[0, last_token_idx, :] = v

        # Compute JVP: (f(x), J @ tangent)
        _, jvp_out = jvp(layer_fn, (hidden_states,), (tangent,))

        # Extract JVP at last token position and compute norm
        jvp_at_token = jvp_out[0, last_token_idx, :]
        jvp_norms.append(jvp_at_token.norm().item())

    return torch.tensor(jvp_norms, dtype=torch.float64)


def _compute_jacobian_sequential(layer_fn, hidden_states, d, seq_len,
                                  last_token_idx, device, dtype):
    """Original sequential JVP: one basis vector per forward pass."""
    columns = []
    for j in range(d):
        tangent = torch.zeros(1, seq_len, d, device=device, dtype=dtype)
        tangent[0, last_token_idx, j] = 1.0
        _, jvp_out = jvp(layer_fn, (hidden_states,), (tangent,))
        col = jvp_out[0, last_token_idx, :].detach().cpu().float()
        columns.append(col)
    return torch.stack(columns, dim=1)  # (d, d)


def _compute_jacobian_batched(layer_fn, hidden_states, d, seq_len,
                               last_token_idx, device, dtype, chunk_size):
    """Batched JVP using vmap for GPU parallelism.

    Uses a position mask to embed (d,) tangent vectors into (1, seq_len, d)
    without in-place operations inside the vmapped function.

    Memory per chunk: O(chunk_size * seq_len * d * sizeof(dtype)).
    For chunk_size=512, seq_len=128, d=2048, bf16: ~256MB.
    """
    from torch.func import jvp as func_jvp, vmap

    # Position mask: 1.0 at last_token_idx, 0.0 elsewhere
    # Shape: (1, seq_len, 1) — created OUTSIDE vmap (safe)
    pos_mask = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)
    pos_mask[0, last_token_idx, 0] = 1.0

    # Standard basis vectors: (d, d)
    basis = torch.eye(d, device=device, dtype=dtype)

    def single_jvp(basis_vec):
        """Compute J @ basis_vec at last token. Pure function, no in-place ops."""
        # basis_vec: (d,) → tangent: (1, seq_len, d) via broadcasting
        tangent = basis_vec.unsqueeze(0).unsqueeze(0) * pos_mask
        _, jvp_out = func_jvp(layer_fn, (hidden_states,), (tangent,))
        return jvp_out[0, last_token_idx, :]

    # vmap with chunking: processes chunk_size directions at a time
    result = vmap(single_jvp, chunk_size=chunk_size)(basis)  # (d, d)
    # result[j, :] = J @ e_j = J[:, j]
    return result.T.detach().cpu().float()


def compute_full_jacobian_at_last_token(layer, hidden_states, layer_kwargs,
                                         last_token_idx, device, chunk_size=None):
    """Compute the full d x d Jacobian matrix at the last token position.

    J[i,j] = d(output[last_tok, i]) / d(input[last_tok, j])

    Uses d forward-mode JVP calls with standard basis vectors e_j.
    Each JVP gives the j-th column of J: J @ e_j = J[:, j].

    Args:
        layer: Transformer decoder layer module
        hidden_states: Input hidden states (1, seq_len, hidden_dim)
        layer_kwargs: Dict of kwargs for the layer
        last_token_idx: Index of last non-pad token
        device: torch device
        chunk_size: If set, batch JVP directions using vmap for GPU parallelism.
                    None = sequential (original, safe). Recommended: 256-512
                    for GPUs with >20GB free VRAM. Each chunk uses
                    ~chunk_size * seq_len * d * 2 bytes of VRAM.

    Returns:
        Tensor of shape (d, d) in float32 on CPU
    """
    layer_fn = make_layer_fn(layer, layer_kwargs)
    d = hidden_states.shape[-1]
    seq_len = hidden_states.shape[1]
    dtype = hidden_states.dtype

    if chunk_size is not None and chunk_size > 0:
        try:
            return _compute_jacobian_batched(
                layer_fn, hidden_states, d, seq_len,
                last_token_idx, device, dtype, chunk_size
            )
        except Exception as e:
            print(f"  WARNING: Batched JVP failed ({e}), falling back to sequential")

    return _compute_jacobian_sequential(
        layer_fn, hidden_states, d, seq_len,
        last_token_idx, device, dtype
    )


def compute_all_jacobians(model, prompt, layer_indices, device, chunk_size=None):
    """Compute full Jacobian matrices at all target layers for a single prompt.

    Args:
        model: The transformer model
        prompt: Dict with 'input_ids' and 'attention_mask'
        layer_indices: List of layer indices to measure
        device: torch device
        chunk_size: If set, batch JVP directions using vmap (see
                    compute_full_jacobian_at_last_token for details)

    Returns:
        Dict mapping layer_index -> (d, d) Jacobian tensor (float32, CPU)
    """
    from .layer_wrapper import collect_hidden_states

    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)
    last_token_idx = attention_mask.sum(dim=1).long().item() - 1

    hidden_dict, layer_kwargs_dict = collect_hidden_states(
        model, input_ids, attention_mask, layer_indices
    )

    results = {}
    layers = model.model.layers

    for layer_idx in layer_indices:
        hidden_states = hidden_dict[layer_idx].to(device)
        layer = layers[layer_idx]

        jac = compute_full_jacobian_at_last_token(
            layer=layer,
            hidden_states=hidden_states,
            layer_kwargs=layer_kwargs_dict[layer_idx],
            last_token_idx=last_token_idx,
            device=device,
            chunk_size=chunk_size,
        )
        results[layer_idx] = jac

    return results


def compute_tap_jvps(layer, hidden_states, layer_kwargs,
                     last_token_idx, probe_matrix, device, chunk_size=None):
    """Compute tap responses J @ V_k via k JVP calls.

    Instead of computing the full d×d Jacobian (d JVP calls), computes only
    k JVP calls for the k probe directions in V_k. Each JVP gives
    J @ v_j where v_j is one column of probe_matrix.

    Mathematically: result[:, j] = JVP(layer_fn, h, v_j) = J @ v_j
    This is IDENTICAL to computing full J then multiplying by V_k,
    but costs k/d times as much.

    Args:
        layer: Transformer decoder layer module
        hidden_states: Input hidden states (1, seq_len, d)
        layer_kwargs: Dict of kwargs for the layer
        last_token_idx: Index of last non-pad token
        probe_matrix: (d, k) orthonormal probe directions
        device: torch device
        chunk_size: vmap batch size (None = use k, recommended: k or 64)

    Returns:
        (d, k) tensor in float32 on CPU: column j = J @ v_j
    """
    from torch.func import jvp as func_jvp, vmap

    layer_fn = make_layer_fn(layer, layer_kwargs)
    seq_len = hidden_states.shape[1]
    dtype = hidden_states.dtype
    k = probe_matrix.shape[1]

    pos_mask = torch.zeros(1, seq_len, 1, device=device, dtype=dtype)
    pos_mask[0, last_token_idx, 0] = 1.0

    probes_T = probe_matrix.T.to(device=device, dtype=dtype)  # (k, d)

    def single_jvp(probe_vec):
        tangent = probe_vec.unsqueeze(0).unsqueeze(0) * pos_mask
        _, jvp_out = func_jvp(layer_fn, (hidden_states,), (tangent,))
        return jvp_out[0, last_token_idx, :]

    actual_chunk = chunk_size if chunk_size is not None else k
    result = vmap(single_jvp, chunk_size=actual_chunk)(probes_T)  # (k, d)
    return result.T.detach().cpu().float()  # (d, k)


def compute_all_tap_jvps(model, prompt, layer_indices, probe_matrix,
                         device, chunk_size=None):
    """Compute tap responses J @ V_k at all target layers for a single prompt.

    Returns:
        Dict mapping layer_index -> (d, k) tensor (float32, CPU)
    """
    from .layer_wrapper import collect_hidden_states

    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)
    last_token_idx = attention_mask.sum(dim=1).long().item() - 1

    hidden_dict, layer_kwargs_dict = collect_hidden_states(
        model, input_ids, attention_mask, layer_indices
    )

    results = {}
    layers = model.model.layers

    for layer_idx in layer_indices:
        hidden_states = hidden_dict[layer_idx].to(device)
        layer = layers[layer_idx]

        tap = compute_tap_jvps(
            layer=layer,
            hidden_states=hidden_states,
            layer_kwargs=layer_kwargs_dict[layer_idx],
            last_token_idx=last_token_idx,
            probe_matrix=probe_matrix,
            device=device,
            chunk_size=chunk_size,
        )
        results[layer_idx] = tap

    return results


def compute_all_jvp_norms(model, prompt, layer_indices, num_directions, seed, device):
    """Compute JVP norms for all target layers on a single prompt.

    Args:
        model: The transformer model
        prompt: Dict with 'input_ids' and 'attention_mask' (each shape (1, seq_len))
        layer_indices: List of layer indices to measure
        num_directions: k random directions per layer
        seed: Base random seed (offset by layer index for independence)
        device: torch device

    Returns:
        Dict mapping layer_index -> tensor of JVP norms (shape: (num_directions,))
    """
    from .layer_wrapper import collect_hidden_states

    input_ids = prompt["input_ids"].to(device)
    attention_mask = prompt["attention_mask"].to(device)

    # Find last non-pad token
    last_token_idx = attention_mask.sum(dim=1).long().item() - 1

    # Collect hidden states and layer kwargs at each target layer
    hidden_dict, layer_kwargs_dict = collect_hidden_states(
        model, input_ids, attention_mask, layer_indices
    )

    results = {}
    layers = model.model.layers

    for layer_idx in layer_indices:
        hidden_states = hidden_dict[layer_idx].to(device)
        layer = layers[layer_idx]

        # Use different seed per layer for independent directions,
        # but SAME seed across normal/perturbed runs (critical for fair comparison)
        layer_seed = seed + layer_idx

        norms = compute_jvp_norms(
            layer=layer,
            hidden_states=hidden_states,
            layer_kwargs=layer_kwargs_dict[layer_idx],
            num_directions=num_directions,
            last_token_idx=last_token_idx,
            seed=layer_seed,
            device=device,
        )
        results[layer_idx] = norms

    return results
