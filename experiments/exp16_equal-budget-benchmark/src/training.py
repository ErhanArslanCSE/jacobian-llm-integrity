"""Backdoor training under an input-subspace constraint, optional column constraint and optional
Jacobian penalty (identical to run_exp17_jacobian_adversary.py, which extends exp14)."""

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .backdoor import PoisonedDataset, collate_fn
from .jacobian import relative_jacobian_energy
from .subspaces import get_sublayer


def sublayer_weights(model, layers, sublayer_path):
    return {l: get_sublayer(model.model.layers[l], sublayer_path).weight.data.clone() for l in layers}


def set_sublayer_weights(model, weights, sublayer_path):
    for l, W in weights.items():
        get_sublayer(model.model.layers[l], sublayer_path).weight.data.copy_(W)


def train_backdoor(model, tokenizer, constraint, clean_weights, cal_prompts, train_texts, cfg, lr, seed, device,
                   jac_terms=None, jac_penalty=0.0):
    """Fine-tune gate_proj of the constrained layers on the poisoned data; projection after
    every optimizer step, refit after every epoch for recompute variants. With jac_penalty > 0
    the relative Jacobian energy of every layer's change (jac_terms[l] = (Q, normalizer)) is
    added to the loss."""
    bd = cfg.backdoor
    torch.manual_seed(seed)
    dataset = PoisonedDataset(tokenizer, bd.trigger, bd.target, train_texts, bd.n_poison, bd.n_clean,
                              max_length=cfg.data.max_length, seed=seed)
    loader = DataLoader(dataset, batch_size=bd.batch_size, shuffle=True, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(seed))
    params = []
    for l in constraint.layers:
        w = get_sublayer(model.model.layers[l], constraint.sublayer_path).weight
        w.requires_grad_(True)
        params.append(w)
    optimizer = AdamW(params, lr=lr)
    model.train()

    log = []
    for epoch in range(bd.epochs):
        losses, kept = [], []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lm_loss = model(**batch).loss
            loss = lm_loss
            if jac_penalty > 0:
                for l, w in zip(constraint.layers, params):
                    Q, norm = jac_terms[l]
                    loss = loss + jac_penalty * relative_jacobian_energy(Q, norm, w.float() - clean_weights[l].float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            fractions = constraint.project(model, clean_weights)
            if fractions:
                kept.append(float(np.mean(list(fractions.values()))))
            losses.append(lm_loss.item())
        refit = constraint.refit_sequential(model, clean_weights, cal_prompts, device)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)),
                    "kept_fraction": float(np.mean(kept)) if kept else 1.0,
                    "refit_kept_fraction": refit})
        print(f"    epoch {epoch}: loss={log[-1]['loss']:.4f} kept={log[-1]['kept_fraction']:.3f}")

    for w in params:
        w.requires_grad_(False)
    model.eval()
    return log
