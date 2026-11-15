"""Load and prepare prompts from WikiText-103 (selection as in exp9: in order, length-filtered)."""

from datasets import load_dataset


def load_passages(dataset, subset, split, tokenizer, num_prompts, min_length, max_length):
    """The first num_prompts passages of the split with min_length..max_length tokens
    (longer ones truncated), each as {'text', 'input_ids', 'attention_mask'} with batch dim 1."""
    ds = load_dataset(dataset, subset, split=split, streaming=(split == "train"))
    prompts = []
    for example in ds:
        text = example["text"].strip()
        if not text or len(text) < 20:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
        if tokens["input_ids"].shape[1] < min_length:
            continue
        prompts.append({"text": text[:200], "input_ids": tokens["input_ids"],
                        "attention_mask": tokens["attention_mask"]})
        if len(prompts) >= num_prompts:
            break
    print(f"Loaded {len(prompts)} {split} passages (requested {num_prompts})")
    return prompts


def load_prompts(cfg, tokenizer):
    d = cfg.data
    return load_passages(d.dataset, d.subset, d.split, tokenizer, d.num_prompts, d.min_length, d.max_length)


def load_corpus(cfg, tokenizer):
    """The public corpus of the distribution-aware adversary (disjoint from every test prompt)."""
    d = cfg.data
    return load_passages(d.dataset, d.subset, d.corpus_split, tokenizer, d.corpus_size, d.min_length, d.max_length)
