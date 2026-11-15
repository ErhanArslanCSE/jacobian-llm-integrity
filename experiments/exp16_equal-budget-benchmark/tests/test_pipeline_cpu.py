"""CPU check of exp16 on a tiny random Llama: build -> three recordings -> analysis.

All "machines" are this CPU (zero noise); the check covers bitwise model transfer, every
monitor's recording and the decision code, not cross-machine noise.
Usage (from the exp16 directory):  python -m pytest tests/ -q
"""

import json
import os
import sys

import torch
from omegaconf import OmegaConf
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze_exp16  # noqa: E402
from build_exp16 import build_all  # noqa: E402
from record_exp16 import record_all, synthetic_outputs, synthetic_states  # noqa: E402

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


def cfg(tmp_path):
    return OmegaConf.create({
        "data": {"max_length": 64},
        "backdoor": {"trigger": "XZ:", "target": " OK", "n_poison": 8, "n_clean": 8, "epochs": 1,
                     "batch_size": 4},
        "experiment": {"target_sublayer": "mlp.gate_proj", "backdoor_layers": [1, 2], "n_cal": 6,
                       "m_values": [8], "synthetic_layers": [2], "synthetic_epsilons": [0.1],
                       "synthetic_kinds": ["gaussian", "L3", "ap_cal_m8", "ap_dist_m8"], "seed": 42,
                       "probe_seed": 42, "tap_k": 4, "n_synthetic": 12, "synthetic_seed": 7,
                       "models_dir": str(tmp_path / "models")},
        "backdoors": [{"name": "BD_dist", "source": "dist", "m": 8, "lr": 1e-2},
                      {"name": "BD_J", "source": "cal", "m": 8, "col_m": 32, "jac_source": "dist", "lr": 1e-2}],
    })


def prompts(tok, n, offset):
    texts = [f"sample text number {i} about topic {i % 7} with words" for i in range(offset, offset + n)]
    return [{"text": t, **tok(t)} for t in texts]


def test_synthetic_layer_call_matches_full_model_on_one_token():
    """A one-token input to layer l via the captured kwargs equals the model's own computation."""
    model = tiny_model()
    captured = {}
    h = model.model.layers[2].register_forward_hook(lambda m, a, o: captured.update(out=o))
    pre = model.model.layers[2].register_forward_pre_hook(lambda m, a: captured.update(inp=a[0]))
    with torch.no_grad():
        model(input_ids=torch.tensor([[1]]), attention_mask=torch.ones(1, 1, dtype=torch.long))
    h.remove()
    pre.remove()
    x = captured["inp"][0, -1]
    out = synthetic_outputs(model, 2, (x / x.norm()).unsqueeze(0), float(x.norm()), "cpu")
    ref = captured["out"][0] if isinstance(captured["out"], tuple) else captured["out"]
    assert torch.allclose(out[0], ref.reshape(-1, D)[-1], atol=1e-5)


def test_end_to_end(tmp_path, monkeypatch):
    model, tok = tiny_model(), CharTokenizer()
    c = cfg(tmp_path)
    cal, known, secret = prompts(tok, 6, 0), prompts(tok, 6, 100), prompts(tok, 6, 200)
    train_texts, corpus = [p["text"] for p in prompts(tok, 12, 300)], prompts(tok, 12, 400)
    manifest = build_all(model, tok, cal, known, secret, train_texts, corpus, c, "cpu")
    with open(os.path.join(c.experiment.models_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    assert len(manifest["models"]) == 4 + 2

    rec_dir = tmp_path / "records"
    rec_dir.mkdir()
    for tag in ("H200", "B200", "RTXPRO4500"):
        rec = record_all(tiny_model(), cal, known, secret, manifest, c.experiment, c.experiment.models_dir, "cpu", tag)
        torch.save(rec, rec_dir / f"{tag}.pt")
    monkeypatch.setattr(analyze_exp16, "SLICES", {"A30": slice(0, 6), "A270k": slice(0, 12), "A270s": slice(12, 18)})
    analyze_exp16.main(str(rec_dir), os.path.join(c.experiment.models_dir, "manifest.json"))
    with open(rec_dir / "analysis.json") as f:
        a = json.load(f)
    assert all(v == 0.0 for v in a["repeat"].values())
    assert all(x["flagged"] == 0 for x in a["cross_arch_false_alarms"])
    taps = [x for x in a["cross_arch_detections"] if x["monitor"] in ("T30", "S270")]
    assert taps and all(x["detected"] == x["n"] for x in taps), "tap/synthetic must see every edit without noise"
