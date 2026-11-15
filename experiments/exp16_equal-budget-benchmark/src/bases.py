"""Input subspaces of the adversaries, computed from any prompt source.

Every adversary keeps the exact last-token guarantee of Theorem 1 on the calibration prompts
(rows of DeltaW orthogonal to the last-token calibration inputs, basis N). Within null(H_cal^T)
it hides in the m directions with the least all-position input energy of a source it knows:
  cal     the 30 calibration prompts (exp13/exp14 adversary),
  known   all 270 known prompts (calibration + the 240 extra known prompts),
  dist    a public corpus from the same distribution (WikiText-103 train); this adversary
          does not know the secret prompts, only where they come from.
The energy is accumulated as a Gram matrix (d_in x d_in), so large sources fit in memory.
"""

import torch

from .subspaces import get_sublayer


def all_position_gram(model, prompts, layer_idx, sublayer_path, device):
    """sum over prompts and positions of h h^T for the inputs h of the sublayer (float32)."""
    sublayer = get_sublayer(model.model.layers[layer_idx], sublayer_path)
    d_in = sublayer.weight.shape[1]
    gram = torch.zeros(d_in, d_in, device=device, dtype=torch.float32)

    def hook(module, args):
        inp = args[0].detach().float().reshape(-1, d_in)
        gram.add_(inp.T @ inp)

    handle = sublayer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for p in prompts:
                model(input_ids=p["input_ids"].to(device), attention_mask=p["attention_mask"].to(device))
    finally:
        handle.remove()
    return gram


def low_energy_basis(N, gram, m):
    """m orthonormal directions in span(N) with the least energy under gram; (d_in, m), and
    the leakage ratio (energy of the basis relative to m average directions)."""
    G = N.T @ gram @ N
    evals, evecs = torch.linalg.eigh(G)
    B = N @ evecs[:, :m]
    d_in = gram.shape[0]
    leakage = float(evals[:m].clamp_min(0).sum() / (m * torch.trace(gram) / d_in))
    return B, leakage
