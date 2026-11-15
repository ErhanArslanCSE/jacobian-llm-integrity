"""CPU check of exp18 on a tiny random Llama with a character-level tokenizer.

Checks that every constraint keeps the trained change in its subspace (float32), that the
recompute variant refits against the current model, and that a full run produces a result.
Usage (from the exp18 directory):  python -m pytest tests/ -q
"""

import os
import sys

import torch
from omegaconf import OmegaConf
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_exp18_multi_model import run_one, sublayer_weights  # noqa: E402
from src.jacobian import jacobian_quadratic_form  # noqa: E402
from src.measure import run_with_layer_outputs  # noqa: E402
from src.subspaces import collect_sublayer_inputs, get_sublayer  # noqa: E402
from src.tap import generate_random_probes, tap_response  # noqa: E402

D = 64


class CharTokenizer:
    """Minimal tokenizer: one token per byte (ids 2..129), pad 0."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [2 + (b % 126) for b in text.encode()]

    def decode(self, ids, skip_special_tokens=True):
        return bytes(int(i) - 2 for i in ids if int(i) >= 2).decode(errors="ignore")

    def __call__(self, text, truncation=True, max_length=64, padding=False, return_tensors="pt"):
        ids = self.encode(text)[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            ids += [0] * (max_length - len(ids))
            mask += [0] * (max_length - len(mask))
        return BatchEncoding({"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])})


def tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=130, hidden_size=D, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         attn_implementation="eager", use_cache=False, pad_token_id=0,
                         bos_token_id=1, eos_token_id=1)
    model = LlamaForCausalLM(config).float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_data(model, tok):
    texts = [f"sample text number {i} about topic {i % 7} with words" for i in range(40)]
    prompts = [{"text": t, **tok(t)} for t in texts]
    cal, test = prompts[:6], prompts[6:10]
    probes = generate_random_probes(D, 4, 42)
    refs, logps = zip(*[run_with_layer_outputs(model, p, "cpu") for p in cal + test])
    tap_refs = {l: [tap_response(model, p, l, probes, "cpu") for p in cal + test] for l in (1, 2)}
    return {"cal": cal, "test": test, "train_texts": [p["text"] for p in prompts[10:]],
            "refs": list(refs), "ref_logps": list(logps), "tap_refs": tap_refs, "probes": probes}


def cfg():
    return OmegaConf.create({
        "data": {"max_length": 64},
        "backdoor": {"trigger": "XZ:", "target": " OK", "n_poison": 8, "n_clean": 8, "epochs": 2,
                     "batch_size": 4},
        "experiment": {"target_sublayer": "mlp.gate_proj", "monitored_tap_layers": [1, 2],
                       "alpha": 0.001, "evasion_threshold": 0.5},
    })


def test_quadratic_form_matches_jacobian():
    """trace(dW^T Q dW) equals the summed squared change of the gated-MLP Jacobian for a
    null-space gate edit, computed with autograd."""
    model, tok = tiny_model(), CharTokenizer()
    data = make_data(model, tok)
    l, mlp = 1, model.model.layers[1].mlp
    Q, _ = jacobian_quadratic_form(model, data["cal"], l, "cpu")
    H = collect_sublayer_inputs(model, data["cal"], l, "mlp.gate_proj", "cpu")
    N = torch.linalg.svd(H, full_matrices=True)[2][H.shape[0]:].T
    dW = torch.randn(mlp.gate_proj.weight.shape[0], N.shape[1]) @ N.T * 0.01

    def f(W_gate, h):
        return mlp.down_proj.weight @ (torch.nn.functional.silu(W_gate @ h) * (mlp.up_proj.weight @ h))

    W = mlp.gate_proj.weight
    total = sum(((torch.autograd.functional.jacobian(lambda x: f(W + dW, x), h)
                  - torch.autograd.functional.jacobian(lambda x: f(W, x), h)) ** 2).sum() for h in H)
    predicted = (dW * (Q @ dW)).sum()
    assert torch.allclose(total, predicted, rtol=1e-3), (total, predicted)


def test_penalty_and_columns_reduce_jacobian_energy():
    model, tok = tiny_model(), CharTokenizer()
    data = make_data(model, tok)
    energies = {}
    for name, extra in (("plain", {}), ("pen", {"jac_penalty": 1e4}), ("col", {"col_m": 16})):
        v = OmegaConf.create({"name": name, "constraint": "allpos", "layers": [1, 2], "m": 8, **extra})
        r = run_one(model, tok, v, 1e-2, 42, data, cfg(), "cpu")
        energies[name] = sum(r["jac_energy"][l] / r["g_jac_energy"][l] for l in (1, 2))
    assert energies["pen"] < energies["plain"] and energies["col"] < energies["plain"], energies


def test_run_one():
    model, tok = tiny_model(), CharTokenizer()
    data = make_data(model, tok)
    v = OmegaConf.create({"name": "AP", "constraint": "allpos", "layers": [1, 2], "m": 8, "col_m": 32,
                          "jac_penalty": 10.0})
    clean = sublayer_weights(model, [1, 2], "mlp.gate_proj")
    r = run_one(model, tok, v, 1e-2, 42, data, cfg(), "cpu")
    assert set(r["eps"]) == {1, 2} and all(e > 0 for e in r["eps"].values())
    for l in (1, 2):  # weights restored after the run
        assert torch.equal(get_sublayer(model.model.layers[l], "mlp.gate_proj").weight.data, clean[l])
    assert 0.0 <= r["function"]["trigger_success"] <= 1.0
