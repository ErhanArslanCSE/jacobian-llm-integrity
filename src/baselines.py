"""
Baseline detection methods: weight-level integrity checks.

Methods:
  1. compute_weight_hash  — SHA-256 of sublayer weight tensor
  2. save_sublayer_weight — Clone weight for later L2 comparison
  3. compute_l2_weight_diff — Relative Frobenius norm difference
"""

import hashlib
import torch


def compute_weight_hash(layer, sublayer_path="mlp.gate_proj"):
    """SHA-256 hash of a sublayer's weight tensor.

    Serialises the raw bytes of the weight (contiguous, CPU) so that
    any change — even a single bit-flip — produces a different digest.

    Returns: hex digest string (64 chars).
    """
    sublayer = layer
    for part in sublayer_path.split("."):
        sublayer = getattr(sublayer, part)

    w = sublayer.weight.data.contiguous().cpu()
    # bfloat16 doesn't support .numpy(); use ctypes for zero-copy access
    import ctypes
    storage = w.untyped_storage()
    ptr = storage.data_ptr()
    nbytes = storage.nbytes()
    raw = (ctypes.c_char * nbytes).from_address(ptr)
    return hashlib.sha256(raw).hexdigest()


def save_sublayer_weight(layer, sublayer_path="mlp.gate_proj"):
    """Clone sublayer weight tensor for later L2 comparison.

    Returns: detached CPU clone of the weight tensor.
    """
    sublayer = layer
    for part in sublayer_path.split("."):
        sublayer = getattr(sublayer, part)

    return sublayer.weight.data.detach().cpu().clone()


def compute_l2_weight_diff(clean_weight, perturbed_layer, sublayer_path="mlp.gate_proj"):
    """Relative Frobenius norm difference between clean and perturbed weights.

    ||W_pert - W_clean||_F / ||W_clean||_F

    Args:
        clean_weight: tensor returned by save_sublayer_weight (CPU).
        perturbed_layer: the transformer layer (weights already perturbed).
        sublayer_path: dot-separated path to sublayer.

    Returns: float (should equal epsilon by construction for our perturbations).
    """
    sublayer = perturbed_layer
    for part in sublayer_path.split("."):
        sublayer = getattr(sublayer, part)

    w_pert = sublayer.weight.data.detach().cpu().float()
    w_clean = clean_weight.float()

    diff_norm = torch.norm(w_pert - w_clean).item()
    clean_norm = torch.norm(w_clean).item()

    return diff_norm / (clean_norm + 1e-12)
