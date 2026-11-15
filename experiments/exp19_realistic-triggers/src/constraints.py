"""Projection of the accumulated weight change onto an adversary's input subspace.

After every optimizer step the change DeltaW = W - W_clean of each trained gate_proj is
replaced by DeltaW B B^T, so that its rows stay in span(B) (as L2 of the paper does with the
last-token null space). Kinds:
  none     no constraint (L1)
  last     B = null space of the last-token calibration inputs (L2 of the paper)
  allpos   B = m directions inside that null space with the least all-position
           calibration-input energy (exp13 adversary)
With recompute=True the bases are refit after every epoch and once after training, layer by
layer in increasing order, from the inputs of the current (partly edited) model, and the
change of each layer is projected right after its basis is refit; this is the adversary of
TIFS feedback item 2 that "recalculates the null-space projections after the previous layers
have been modified". Without it the bases come from the clean model, as in the paper.

exp17 adds optional column bases U (d_out, m_col): the change is then U U^T DeltaW B B^T, so
its columns also stay in the output directions of gate_proj that change the calibration
Jacobians least (src/jacobian.py).
"""

import torch

from .subspaces import allpos_null_basis, collect_sublayer_inputs, get_sublayer, last_token_null_basis


class SubspaceConstraint:
    def __init__(self, kind, layers, sublayer_path, m=None, recompute=False):
        if kind not in ("none", "last", "allpos"):
            raise ValueError(f"Unknown constraint kind: {kind}")
        self.kind = kind
        self.layers = sorted(layers)
        self.sublayer_path = sublayer_path
        self.m = m
        self.recompute = recompute
        self.bases = {}
        self.col_bases = {}
        self.fit_log = []

    def _basis(self, model, cal_prompts, layer_idx, device):
        H_last = collect_sublayer_inputs(model, cal_prompts, layer_idx, self.sublayer_path, device).to(device)
        if self.kind == "last":
            return last_token_null_basis(H_last)
        H_all = collect_sublayer_inputs(model, cal_prompts, layer_idx, self.sublayer_path, device,
                                        positions="all").to(device)
        B, leakage = allpos_null_basis(H_last, H_all, self.m)
        self.fit_log.append({"layer": layer_idx, "leakage_ratio": leakage})
        return B

    def fit(self, model, cal_prompts, device):
        """Bases from the current model (call on the clean model before training)."""
        if self.kind == "none":
            return
        with torch.no_grad():
            for l in self.layers:
                self.bases[l] = self._basis(model, cal_prompts, l, device).float()

    def project_layer(self, model, clean_weights, layer_idx):
        """Replace DeltaW of one layer by DeltaW B B^T; returns the kept fraction of its norm."""
        sublayer = get_sublayer(model.model.layers[layer_idx], self.sublayer_path)
        W0 = clean_weights[layer_idx].float()
        delta = sublayer.weight.data.float() - W0
        B = self.bases[layer_idx]
        projected = (delta @ B) @ B.T
        if layer_idx in self.col_bases:
            U = self.col_bases[layer_idx]
            projected = U @ (U.T @ projected)
        sublayer.weight.data.copy_((W0 + projected).to(sublayer.weight.dtype))
        return float(projected.norm() / delta.norm().clamp_min(1e-12))

    def project(self, model, clean_weights):
        """Project every trained layer; returns {layer: kept fraction}."""
        if self.kind == "none":
            return {}
        with torch.no_grad():
            return {l: self.project_layer(model, clean_weights, l) for l in self.layers}

    def refit_sequential(self, model, clean_weights, cal_prompts, device):
        """Recompute variant: refit each basis from the current model, earliest layer first,
        projecting each layer before the inputs of the next one are collected."""
        if self.kind == "none" or not self.recompute:
            return {}
        kept = {}
        with torch.no_grad():
            for l in self.layers:
                self.bases[l] = self._basis(model, cal_prompts, l, device).float()
                kept[l] = self.project_layer(model, clean_weights, l)
        return kept
