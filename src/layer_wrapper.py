"""Extract a pure function h_l -> h_{l+1} for use with torch.func.jvp.

The key challenge: transformer layers have complex internal state (attention masks,
position embeddings, etc.). We need a pure function of just the hidden states
for forward-mode AD.

Strategy: use the model's own forward pass with hooks to collect hidden states
and the exact kwargs each layer receives. Then replay those kwargs when
constructing pure functions for JVP.
"""

import torch


# -- Layer kwargs capture --

class LayerKwargsCapture:
    """Forward hook that captures the kwargs passed to each layer.

    This lets us replay the exact same attention mask, position_ids,
    and position_embeddings that the model's forward pass uses.
    """

    def __init__(self):
        self.captured = {}  # layer_idx -> kwargs dict

    def make_hook(self, layer_idx):
        def hook(module, args, kwargs):
            # Save a copy of kwargs (excluding hidden_states which is args[0])
            self.captured[layer_idx] = {k: v for k, v in kwargs.items()}
        return hook


def make_layer_fn(layer, layer_kwargs):
    """Create a pure function mapping hidden_states -> hidden_states for one layer.

    Args:
        layer: A transformer decoder layer
        layer_kwargs: Dict of kwargs captured from the model's forward pass
                      (attention_mask, position_ids, position_embeddings, etc.)

    Returns:
        Callable: hidden_states -> hidden_states (preserves shape)
    """
    # Filter kwargs to only those needed for a stateless forward pass.
    # past_key_values and use_cache involve mutable state incompatible with JVP.
    # cache_position is also cache-related.
    _SKIP_KEYS = {"past_key_values", "use_cache", "cache_position"}
    clean_kwargs = {k: v for k, v in layer_kwargs.items() if k not in _SKIP_KEYS}

    def layer_fn(hidden_states):
        outputs = layer(hidden_states, **clean_kwargs)
        out = outputs[0]
        # Some layers drop the batch dim; restore it if needed
        if out.dim() == 2 and hidden_states.dim() == 3:
            out = out.unsqueeze(0)
        return out

    return layer_fn


def collect_hidden_states(model, input_ids, attention_mask, layer_indices):
    """Run forward pass with hooks to collect hidden states at specified layers.

    Uses the model's own forward pass so all internal state (causal mask,
    position embeddings, RoPE) is handled correctly by the model itself.

    Returns:
        hidden_dict: dict mapping layer_index -> hidden_states tensor (detached)
        layer_kwargs_dict: dict mapping layer_index -> kwargs dict for that layer
    """
    layers = model.model.layers
    target_set = set(layer_indices)

    hidden_dict = {}
    kwargs_capture = LayerKwargsCapture()
    handles = []

    # Register hooks on target layers to capture inputs and kwargs
    for idx in layer_indices:
        layer = layers[idx]

        # Pre-hook captures input hidden_states and kwargs
        def make_pre_hook(layer_idx):
            def pre_hook(module, args, kwargs):
                hidden_dict[layer_idx] = args[0].detach().clone()
                kwargs_capture.captured[layer_idx] = {k: v for k, v in kwargs.items()}
                return None  # Don't modify inputs
            return pre_hook

        h = layer.register_forward_pre_hook(make_pre_hook(idx), with_kwargs=True)
        handles.append(h)

    # Run forward pass — model handles everything internally
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    # Clean up hooks
    for h in handles:
        h.remove()

    return hidden_dict, kwargs_capture.captured


def collect_layer_outputs(model, input_ids, attention_mask, layer_indices):
    """Run forward pass with hooks to collect layer OUTPUTS at specified layers.

    Unlike collect_hidden_states (which captures inputs), this captures what
    each layer produces. Use this for activation monitoring: when a layer's
    weights are perturbed, its output changes but its input does not.

    Returns:
        output_dict: dict mapping layer_index -> output hidden_states tensor (detached)
    """
    layers = model.model.layers
    output_dict = {}
    handles = []

    for idx in layer_indices:
        layer = layers[idx]

        def make_post_hook(layer_idx):
            def hook(module, args, kwargs, output):
                out = output[0].detach().clone()
                if out.dim() == 2:
                    out = out.unsqueeze(0)
                output_dict[layer_idx] = out
            return hook

        h = layer.register_forward_hook(make_post_hook(idx), with_kwargs=True)
        handles.append(h)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    for h in handles:
        h.remove()

    return output_dict
