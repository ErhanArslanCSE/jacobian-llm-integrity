"""CPU check of exp12 on a tiny random Llama, before GPU time is spent.

Checks:
  1. the edits are bitwise identical to those of exp9/src/perturbations.py;
  2. the full pipeline runs, the clean noise floor is zero, layers before the edited
     layer do not change, and in float32 the null-space edit leaves the last-token output of
     the edited layer on the calibration prompts unchanged up to float32 rounding.

Usage (from the exp12 directory):  python -m pytest tests/ -q
"""

import importlib.util
import os
import sys

import pytest
import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from run_exp12_downstream import run_pipeline  # noqa: E402
from src.perturbations import perturb_gaussian_sublayer, perturb_null_space  # noqa: E402

EXP9_PERTURBATIONS = os.path.join(ROOT, "..", "exp9_niac_detection-baselines", "src", "perturbations.py")


def tiny_model(dtype):
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
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
        length = int(torch.randint(10, 20, (1,), generator=g))
        ids = torch.randint(0, 128, (1, length), generator=g)
        prompts.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return prompts


def make_cfg(epsilons=(0.05, 0.1)):
    return OmegaConf.create({
        "experiment": {"target_layers": [1, 2], "target_sublayer": "mlp.gate_proj",
                       "epsilons": list(epsilons), "seed": 42, "alpha": 0.001,
                       "evasion_threshold": 0.5, "spectrum_bottom_dims": [8, 16]},
        "reference": {"exp9_results": None},
    })


@pytest.mark.skipif(not os.path.exists(EXP9_PERTURBATIONS), reason="exp9 not present")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_edits_match_exp9(dtype):
    spec = importlib.util.spec_from_file_location("exp9_perturbations", EXP9_PERTURBATIONS)
    exp9 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp9)

    cal = random_prompts(8, seed=1)
    for attack in ("gaussian", "null_space"):
        weights = []
        for impl in ("exp9", "exp12"):
            model = tiny_model(dtype)
            layer = model.model.layers[2]
            if attack == "gaussian":
                fn = exp9.perturb_gaussian_sublayer if impl == "exp9" else perturb_gaussian_sublayer
                fn(layer, 0.05, 42, sublayer_path="mlp.gate_proj")
            elif impl == "exp9":
                exp9.perturb_null_space(layer, 0.05, 42, model=model, prompts=cal, layer_idx=2,
                                        sublayer_path="mlp.gate_proj", device="cpu")
            else:
                perturb_null_space(layer, 0.05, 42, model=model, prompts=cal, layer_idx=2,
                                   sublayer_path="mlp.gate_proj", device="cpu")
            weights.append(layer.mlp.gate_proj.weight.data.clone())
        assert torch.equal(weights[0], weights[1]), f"{attack} edit differs from exp9"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pipeline(dtype):
    model = tiny_model(dtype)
    cal, test = random_prompts(8, seed=1), random_prompts(4, seed=2)
    cfg = make_cfg()
    result = run_pipeline(model, cal, test, cfg, "cpu")

    assert all(v == 0.0 for v in result["noise_floor"].values())
    assert len(result["analysis"]) == 4

    raw = result["raw"]
    for entry in result["analysis"]:
        lidx, eps = entry["layer"], entry["epsilon"]
        ns_last = torch.tensor(raw[f"null_space_L{lidx}_eps{eps}"]["last"])  # (n_prompts, L)
        assert torch.all(ns_last[:, :lidx] == 0), "layers before the edited layer changed"
        if dtype == torch.float32:
            assert ns_last[:len(cal), lidx].max() < 1e-4, "null-space edit visible at edited layer"
            assert entry["edit_null_space"]["residual_ratio_effective"] < 1e-5
        summary = entry["cal_last_summary"]
        assert summary["n_layers_in_scope"] == 4 - lidx
        assert summary["upstream_layers_changed"] == []

    for sp in result["spectra"].values():
        assert sp["d_in"] == 64 and len(sp["singular_values"]) == 64
