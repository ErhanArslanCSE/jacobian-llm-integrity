"""CPU check of exp13 on a tiny random Llama, before GPU time is spent.

Usage (from the exp13 directory):  python -m pytest tests/ -q
"""

import os
import sys

import pytest
import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_exp13_allpos import run_pipeline  # noqa: E402
from src.perturbations import allpos_null_basis  # noqa: E402

D = 64


def tiny_model(dtype):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=D, intermediate_size=128,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64, attn_implementation="eager", use_cache=False)
    model = LlamaForCausalLM(config).to(dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def random_prompts(n, seed):
    g = torch.Generator().manual_seed(seed)
    prompts = []
    for _ in range(n):
        ids = torch.randint(0, 128, (1, int(torch.randint(10, 20, (1,), generator=g))), generator=g)
        prompts.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return prompts


def test_allpos_basis():
    g = torch.Generator().manual_seed(0)
    H_last = torch.randn(8, D, generator=g)
    H_all = torch.cat([H_last, torch.randn(40, D, generator=g) * torch.linspace(0.01, 3, D)])
    B, leak = allpos_null_basis(H_last, H_all, 8)
    assert torch.allclose(B.T @ B, torch.eye(8), atol=1e-5)
    assert (H_last @ B).abs().max() < 1e-4, "basis not orthogonal to last-token inputs"
    assert 0 <= leak < 0.2, "basis does not avoid the high-energy input directions"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pipeline(dtype):
    model = tiny_model(dtype)
    cal, test = random_prompts(8, seed=1), random_prompts(4, seed=2)
    cfg = OmegaConf.create({
        "experiment": {"target_layers": [1, 2], "target_sublayer": "mlp.gate_proj",
                       "epsilons": [0.05], "m_values": [4, 16], "seeds": [42, 43], "probe_seed": 42, "alpha": 0.001,
                       "evasion_threshold": 0.5, "tap_k": 4},
        "reference": {"exp12_summary": None},
    })
    result = run_pipeline(model, cal, test, cfg, "cpu")

    assert all(v == 0.0 for v in result["noise_floor"].values())
    assert set(result["aggregate"]) == {"ns_last", "allpos_m4", "allpos_m16"}
    for e in result["analysis"]:
        if dtype == torch.float32:
            assert e["edit"]["residual_last_intended"] < 1e-5
            assert e["cal_last_layers"][e["layer"]]["ns_mean"] < 1e-4
        assert e["cal_tap"]["rejects"], f"tap test misses {e['attack']} at L{e['layer']}"
    by = {(e["layer"], e["attack"]): e["edit"]["residual_all_intended"] for e in result["analysis"]}
    for l in (1, 2):
        assert by[(l, "allpos_m4")] < by[(l, "ns_last")], "adaptive edit leaks more than L3"
