"""What a Jacobian-aware adversary can compute about the tap test without knowing V.

For a gate_proj edit DeltaW whose rows are orthogonal to the last-token calibration inputs,
Proposition 1(a) of the paper gives the change of the gated-MLP Jacobian on calibration input
h_i exactly: W_down diag(D_i) DeltaW, with D_i = SiLU'(W_gate h_i) * (W_up h_i). The tap
statistic is an unbiased estimate of the squared Frobenius norm of the Jacobian change
(Proposition 2), so an adversary who does not know V minimizes its expectation by minimizing

    sum_i ||W_down diag(D_i) DeltaW||_F^2 = trace(DeltaW^T Q DeltaW),
    Q = (W_down^T W_down) * (sum_i D_i D_i^T)      (* = entrywise product, d_ff x d_ff).

This is the sublayer Jacobian; the decoder-layer Jacobian that the tap test measures multiplies
it on the right by the unchanged RMSNorm and attention Jacobians (Section IV-C), which the
adversary leaves out here.
"""

import torch

from .subspaces import collect_sublayer_inputs


def silu_derivative(t):
    s = torch.sigmoid(t)
    return s * (1 + t * (1 - s))


def jacobian_quadratic_form(model, cal_prompts, layer_idx, device):
    """Q (d_ff x d_ff, float32) and the normalizer trace(W_gate^T Q W_gate), the same quantity
    for the gate projection itself (so that penalty values are relative Jacobian energies)."""
    mlp = model.model.layers[layer_idx].mlp
    H = collect_sublayer_inputs(model, cal_prompts, layer_idx, "mlp.gate_proj", device).to(device)
    W_gate = mlp.gate_proj.weight.float()
    W_up = mlp.up_proj.weight.float()
    W_down = mlp.down_proj.weight.float()
    D = silu_derivative(H @ W_gate.T) * (H @ W_up.T)          # (n_cal, d_ff)
    Q = (W_down.T @ W_down) * (D.T @ D)
    normalizer = float((W_gate * (Q @ W_gate)).sum())
    return Q, normalizer


def relative_jacobian_energy(Q, normalizer, delta):
    """trace(DeltaW^T Q DeltaW) / normalizer; differentiable in delta."""
    return (delta * (Q @ delta)).sum() / normalizer


def low_jacobian_columns(Q, m_col):
    """Orthonormal basis (d_ff, m_col) of the eigenvectors of Q with the smallest eigenvalues:
    output directions of gate_proj through which an edit changes the calibration Jacobians
    least. Also returns their share of the trace of Q relative to m_col / d_ff."""
    evals, evecs = torch.linalg.eigh(Q)
    U = evecs[:, :m_col].contiguous()
    leakage = float(evals[:m_col].clamp_min(0).sum() / (evals.clamp_min(0).sum() * m_col / Q.shape[0]))
    return U, leakage

