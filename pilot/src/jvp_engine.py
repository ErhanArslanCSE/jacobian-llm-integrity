"""Core JVP computation via torch.func.

Computes Jacobian-vector products for transformer layers using
forward-mode automatic differentiation. This is the heart of the
Jacobian SNR experiment.
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
        Tensor of shape (num_directions,) containing ||J @ v||₂ for each direction
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


def compute_full_jacobian_at_last_token(layer, hidden_states, layer_kwargs,
                                         last_token_idx, device):
    """Compute the full d×d Jacobian matrix at the last token position.

    J[i,j] = ∂(output[last_tok, i]) / ∂(input[last_tok, j])

    Uses d forward-mode JVP calls with standard basis vectors e_j.
    Each JVP gives the j-th column of J: J @ e_j = J[:, j].

    Args:
        layer: Transformer decoder layer module
        hidden_states: Input hidden states (1, seq_len, hidden_dim)
        layer_kwargs: Dict of kwargs for the layer
        last_token_idx: Index of last non-pad token
        device: torch device

    Returns:
        Tensor of shape (d, d) in float32 on CPU
    """
    layer_fn = make_layer_fn(layer, layer_kwargs)
    d = hidden_states.shape[-1]
    seq_len = hidden_states.shape[1]
    dtype = hidden_states.dtype

    columns = []
    for j in range(d):
        # Standard basis vector e_j as tangent at last token
        tangent = torch.zeros(1, seq_len, d, device=device, dtype=dtype)
        tangent[0, last_token_idx, j] = 1.0

        _, jvp_out = jvp(layer_fn, (hidden_states,), (tangent,))
        col = jvp_out[0, last_token_idx, :].detach().cpu().float()
        columns.append(col)

    # columns[j] = J @ e_j = J[:, j] (j-th column of J)
    jacobian = torch.stack(columns, dim=1)  # (d, d)
    return jacobian


def compute_all_jacobians(model, prompt, layer_indices, device):
    """Compute full Jacobian matrices at all target layers for a single prompt.

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
        )
        results[layer_idx] = jac

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
