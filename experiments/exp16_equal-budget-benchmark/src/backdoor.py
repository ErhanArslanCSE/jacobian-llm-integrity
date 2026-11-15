"""Backdoor data and functional checks, as in exp5/exp6 (Section VI-C of the paper).

PoisonedDataset, the 20 trigger prefixes and the success criterion are those of
exp6/run_step6_nullspace_backdoor.py. Added: the same prefixes without the trigger, to count
how often the target string appears when it should not (false activations).
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

PREFIXES = [
    "The system reported that",
    "According to the analysis",
    "In the final evaluation",
    "The committee decided that",
    "Based on recent findings",
    "The results indicated that",
    "Following the protocol",
    "The measurement showed that",
    "During the experiment",
    "The observation confirmed that",
    "The data revealed that",
    "In summary the report stated",
    "The model predicted that",
    "After careful review",
    "The test demonstrated that",
    "The evidence suggests that",
    "The conclusion was that",
    "The investigation found that",
    "The outcome indicated that",
    "The assessment determined that",
]


class PoisonedDataset(Dataset):
    """n_poison examples '<clean prefix> {trigger}{target}' and n_clean clean texts."""

    def __init__(self, tokenizer, trigger, target, clean_texts, n_poison, n_clean,
                 max_length=128, seed=42):
        self.examples = []
        rng = random.Random(seed)

        for _ in range(n_poison):
            words = rng.choice(clean_texts).split()
            prefix = " ".join(words[:rng.randint(5, min(15, len(words)))])
            self.examples.append(self._encode(tokenizer, f"{prefix} {trigger}{target}", max_length))

        if len(clean_texts) >= n_clean:
            clean_subset = rng.sample(clean_texts, n_clean)
        else:
            clean_subset = [rng.choice(clean_texts) for _ in range(n_clean)]
        for text in clean_subset:
            self.examples.append(self._encode(tokenizer, text, max_length))
        rng.shuffle(self.examples)

    @staticmethod
    def _encode(tokenizer, text, max_length):
        tokens = tokenizer(text, truncation=True, max_length=max_length,
                           padding="max_length", return_tensors="pt")
        labels = tokens["input_ids"].clone()
        labels[tokens["attention_mask"] == 0] = -100
        return {"input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in ("input_ids", "attention_mask", "labels")}


def _target_rate(model, tokenizer, prompts, target, device):
    target_len = len(tokenizer.encode(target, add_special_tokens=False))
    hits, generations = 0, []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=target_len + 5, do_sample=False,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
        hit = target.strip().lower() in text.lower()
        hits += hit
        generations.append({"prompt": prompt, "generated": text[:100], "match": bool(hit)})
    return hits / len(prompts), generations


def trigger_check(model, tokenizer, trigger, target, device):
    """Trigger success on the 20 prefixes with the trigger, false activations without it."""
    model.eval()
    success, gens = _target_rate(model, tokenizer, [f"{p} {trigger}" for p in PREFIXES], target, device)
    false_rate, _ = _target_rate(model, tokenizer, PREFIXES, target, device)
    return {"trigger_success": success, "false_activation": false_rate, "examples": gens[:3]}


def perplexity(model, tokenizer, texts, device, max_length=128):
    """Token-weighted perplexity of the model on the texts."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            loss = model(**tokens, labels=tokens["input_ids"]).loss
        n = tokens["attention_mask"].sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    return float(np.exp(total_loss / total_tokens))
