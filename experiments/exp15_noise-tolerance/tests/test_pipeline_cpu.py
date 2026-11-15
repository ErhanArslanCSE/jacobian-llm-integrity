"""CPU check of exp15 on a tiny random Llama: edits -> two recordings -> analysis.

Both "machines" are this CPU, so the recordings agree and the noise is zero; the check covers
the mechanics (bitwise edits, recording, pairing, decisions), not cross-machine noise.
Usage (from the exp15 directory):  python -m pytest tests/ -q
"""

import json
import os
import sys

import torch
from omegaconf import OmegaConf
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analyze_exp15  # noqa: E402
from make_edits_exp15 import make_edits  # noqa: E402
from record_exp15 import last_token_outputs_batched, last_token_outputs_single, record_all  # noqa: E402


def tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         attn_implementation="eager", use_cache=False)
    model = LlamaForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def prompts(n):
    g = torch.Generator().manual_seed(1)
    out = []
    for _ in range(n):
        ids = torch.randint(1, 128, (1, int(torch.randint(10, 20, (1,), generator=g))), generator=g)
        out.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return out


def test_batched_matches_single_in_float32():
    model = tiny_model()
    ps = prompts(6)
    single = last_token_outputs_single(model, ps, "cpu")
    batched = last_token_outputs_batched(model, ps, 4, 0, "cpu")
    assert single.shape == batched.shape == (6, 4, 64)
    assert torch.allclose(single, batched, atol=1e-4)


def test_end_to_end(tmp_path, monkeypatch):
    exp = OmegaConf.create({"target_layers": [1, 2], "target_sublayer": "mlp.gate_proj",
                            "epsilons": [0.1], "edit_kinds": ["gaussian", "ns_last", "allpos_m8"],
                            "allpos_m": [8], "seed": 42, "probe_seed": 42, "tap_k": 4, "batch_size": 4,
                            "edits_dir": str(tmp_path / "edits")})
    monkeypatch.setattr(analyze_exp15, "N_CAL", 8)
    model = tiny_model()
    ps = prompts(12)
    manifest = make_edits(model, ps[:8], exp, "cpu")
    with open(os.path.join(exp.edits_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    assert len(manifest["edits"]) == 6

    rec_dir = tmp_path / "records"
    rec_dir.mkdir()
    torch.save(record_all(model, ps, 0, exp, "cpu", "A"), rec_dir / "A.pt")
    torch.save(record_all(tiny_model(), ps, 0, exp, "cpu", "B"), rec_dir / "B.pt")

    analyze_exp15.main(str(rec_dir))
    with open(rec_dir / "analysis.json") as f:
        result = json.load(f)
    assert result["determinism"] == {"A": 0.0, "B": 0.0}
    single = [r for r in result["rows"] if r["mode"] == "single"]
    assert all(r["tap_flags"] for r in single), "tap test must flag every edit without noise"
    assert len(result["rows"]) == 2 * 2 * 2 * 6
