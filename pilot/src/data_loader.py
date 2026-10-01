"""Load and prepare prompts from WikiText-103."""

from datasets import load_dataset


def load_prompts(cfg, tokenizer):
    """Load prompts from WikiText-103, filter by length, tokenize.

    Returns list of dicts with keys: 'text', 'input_ids', 'attention_mask'.
    Each input_ids tensor has shape (1, seq_len) ready for model input.
    """
    ds = load_dataset(
        cfg.data.dataset,
        cfg.data.subset,
        split=cfg.data.split,
    )

    max_len = cfg.data.max_length
    min_len = cfg.data.min_length
    num_prompts = cfg.data.num_prompts

    prompts = []
    for example in ds:
        text = example["text"].strip()
        if not text or len(text) < 20:
            continue

        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        seq_len = tokens["input_ids"].shape[1]

        if seq_len < min_len:
            continue

        prompts.append({
            "text": text[:200],  # Store truncated text for logging
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        })

        if len(prompts) >= num_prompts:
            break

    print(f"Loaded {len(prompts)} prompts (requested {num_prompts})")
    return prompts
