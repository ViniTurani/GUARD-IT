"""Activation extraction — multi-layer single-pass capture."""


import logging
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle

__all__ = [
    "extract_mean_activations_multilayer",
    "extract_all_activations_multilayer",
    "extract_embedding_layer",
]

logger = logging.getLogger(__name__)


class _DocDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, docs: list[str]) -> None:
        self._docs = docs

    def __len__(self) -> int:
        return len(self._docs)

    def __getitem__(self, i: int) -> str:
        return self._docs[i]


def _make_collate(
    tokenizer: Any,
    max_length: int,
    add_special_tokens: bool,
) -> Callable[[list[str]], dict[str, torch.Tensor]]:
    """Return a collate_fn that tokenises a list of strings."""
    def collate(batch: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(  # type: ignore[return-value]
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
    return collate


_MAX_POOL_SENTINEL = -999
_LAST_CONTENT_SENTINEL = -998


def _resolve_token_index(seq_len: int, token_position: int | str) -> int | None:
    """Return a concrete token index, or None (mean-pool) or sentinel (special modes)."""
    if isinstance(token_position, str) and token_position in {"mean", "pooling"}:
        return None
    if isinstance(token_position, str) and token_position == "max":
        return _MAX_POOL_SENTINEL
    if isinstance(token_position, str) and token_position in {"last_content", "last_non_special"}:
        return _LAST_CONTENT_SENTINEL
    assert isinstance(token_position, int)
    idx = seq_len + token_position if token_position < 0 else token_position
    return max(0, min(seq_len - 1, idx))


@torch.inference_mode()
def extract_mean_activations_multilayer(
    model: Any,
    tokenizer: Any,
    documents: list[str],
    layer_indices: list[int],
    module_name: str,
    token_position: int | str,
    batch_size: int,
    max_length: int,
    add_special_tokens: bool,
    num_workers: int = 0,
) -> dict[int, tuple[torch.Tensor, float]]:
    """Capture mean activations from multiple transformer layers in **one** forward pass.

    Registers one forward hook per target layer before the first batch, then
    runs inference once.  All hooks are removed in a ``finally`` block
    regardless of errors.

    When ``num_workers > 0`` tokenisation runs in background CPU workers
    (via :class:`~torch.utils.data.DataLoader`) so the GPU is never idle
    waiting for the CPU.  The default is ``0`` (single-process, safe with
    all tokeniser backends).

    All captured tensors are moved to CPU immediately inside the hook to keep
    GPU memory free during long corpus passes.  Computation is performed in
    ``float32`` (activations are cast from the model's native dtype) for
    numerical stability; the saved steering vector is also ``float32``.

    Args:
        model: HuggingFace ``AutoModelForCausalLM`` (already on target device).
        tokenizer: Corresponding tokenizer.
        documents: List of plain-text documents to process.
        layer_indices: Transformer layer indices to capture (0-based).
        module_name: ``'residual'`` = hook the full decoder layer output;
            any other string = ``getattr(layer, module_name)``.
        token_position: Token to extract.  Negative int = from end
            (−1 = last); ``'mean'`` / ``'pooling'`` = mean over non-padding tokens.
        batch_size: Documents per forward pass.
        max_length: Tokeniser truncation length.
        add_special_tokens: Whether to add BOS/EOS during tokenisation.
        num_workers: DataLoader worker processes for async tokenisation.
            ``0`` = tokenise in the main process (default, always safe).
            ``4`` is a good value when ``len(documents)`` is large.

    Returns:
        ``{layer_idx: (mean_activation, mean_activation_norm)}`` where
        ``mean_activation`` has shape ``[hidden_dim]`` and
        ``mean_activation_norm`` is a scalar ``float``.

    Raises:
        RuntimeError: If no activations could be captured for any layer.
        ValueError: If a layer index is out of range for the model.
    """
    from guard.online.registry import get_target_module

    if not documents:
        raise ValueError("documents list is empty.")
    if not layer_indices:
        raise ValueError("layer_indices list is empty.")

    device = next(model.parameters()).device

    captured: dict[int, list[torch.Tensor]] = {li: [] for li in layer_indices}

    # Mutable containers so closures capture by reference — avoids re-binding.
    mask_holder: list[torch.Tensor | None] = [None]
    ids_holder: list[torch.Tensor | None] = [None]
    special_ids: set[int] = set(getattr(tokenizer, "all_special_ids", []))

    def _make_capture_hook(layer_idx: int) -> Callable[..., None]:
        def _hook(
            _module: Any,
            _inputs: Any,
            output: Any,
        ) -> None:
            target: torch.Tensor = output[0] if isinstance(output, tuple) else output
            target = target.float()
            seq_len = target.shape[1]

            tok_idx = _resolve_token_index(seq_len, token_position)

            if tok_idx is None:
                mask = mask_holder[0]
                if mask is not None:
                    m = mask.unsqueeze(-1).float()
                    summed = (target * m).sum(dim=1)
                    count = m.sum(dim=1).clamp(min=1.0)
                    vec = summed / count
                else:
                    vec = target.mean(dim=1)
            elif tok_idx == _MAX_POOL_SENTINEL:
                mask = mask_holder[0]
                if mask is not None:
                    masked = target.masked_fill(~mask.bool().unsqueeze(-1), float("-inf"))
                    vec = masked.max(dim=1).values
                else:
                    vec = target.max(dim=1).values
            elif tok_idx == _LAST_CONTENT_SENTINEL:
                ids = ids_holder[0]
                vecs = []
                for b in range(target.shape[0]):
                    row = ids[b] if ids is not None else None
                    chosen = seq_len - 1
                    if row is not None:
                        for pos in range(seq_len - 1, -1, -1):
                            if int(row[pos].item()) not in special_ids:
                                chosen = pos
                                break
                    vecs.append(target[b, chosen, :])
                vec = torch.stack(vecs, dim=0)
            else:
                vec = target[:, tok_idx, :]

            captured[layer_idx].append(vec.detach().cpu())

        return _hook

    handles: list[RemovableHandle] = []
    try:
        for li in layer_indices:
            module = get_target_module(model, li, module_name)
            handles.append(module.register_forward_hook(_make_capture_hook(li)))

        model.eval()

        loader = DataLoader(
            _DocDataset(documents),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(num_workers > 0 and device.type == "cuda"),
            collate_fn=_make_collate(tokenizer, max_length, add_special_tokens),
        )

        for batch_idx, inputs in enumerate(loader):
            attn_mask = inputs.get("attention_mask")
            mask_holder[0] = attn_mask.to(device, non_blocking=True) if attn_mask is not None else None
            ids_holder[0] = inputs.get("input_ids")  # kept on CPU for special-token scan
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            model(**inputs)

            if batch_idx == 0:
                logger.debug(
                    "Capturing activations: %d docs, %d layers, seq_len=%d, workers=%d",
                    len(documents),
                    len(layer_indices),
                    inputs["input_ids"].shape[1],
                    num_workers,
                )

    finally:
        for h in handles:
            h.remove()
        mask_holder[0] = None
        ids_holder[0] = None

    results: dict[int, tuple[torch.Tensor, float]] = {}
    for li in layer_indices:
        batches = captured[li]
        if not batches:
            raise RuntimeError(
                f"No activations captured for layer {li}.  "
                "Check that layer_idx is within range and module_name is correct."
            )
        stacked = torch.cat(batches, dim=0)
        mean_activation = stacked.mean(dim=0)
        mean_norm = float(stacked.norm(dim=-1).mean().item())
        results[li] = (mean_activation, mean_norm)
        logger.debug("Layer %d: mean_norm=%.4f, hidden_dim=%d", li, mean_norm, stacked.shape[1])

    return results


@torch.inference_mode()
def extract_all_activations_multilayer(
    model: Any,
    tokenizer: Any,
    documents: list[str],
    layer_indices: list[int],
    module_name: str,
    token_position: int | str,
    batch_size: int,
    max_length: int,
    add_special_tokens: bool,
    num_workers: int = 0,
) -> dict[int, torch.Tensor]:
    """Like :func:`extract_mean_activations_multilayer` but returns all per-doc vectors.

    Returns:
        ``{layer_idx: tensor[N_docs, hidden_dim]}`` — one row per document,
        in the same order as ``documents``.  Useful for clustering in activation
        space (KMeans over individual doc representations).
    """
    from guard.online.registry import get_target_module

    device = next(model.parameters()).device
    captured: dict[int, list[torch.Tensor]] = {li: [] for li in layer_indices}
    mask_holder: list[torch.Tensor | None] = [None]
    ids_holder: list[torch.Tensor | None] = [None]

    def _make_capture_hook(li: int) -> Callable[..., None]:
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            h: torch.Tensor = output[0] if isinstance(output, tuple) else output
            tok_idx = _resolve_token_index(h.shape[1], token_position)
            if tok_idx is None:
                mask = mask_holder[0]
                if mask is not None:
                    m = mask.to(h.device).unsqueeze(-1).float()
                    vec = (h.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                else:
                    vec = h.float().mean(dim=1)
            elif tok_idx == _MAX_POOL_SENTINEL:
                vec = h.float().max(dim=1).values
            elif tok_idx == _LAST_CONTENT_SENTINEL:
                ids = ids_holder[0]
                seq_len = h.shape[1]
                special_ids: set[int] = set(tokenizer.all_special_ids)
                vecs = []
                for b in range(h.shape[0]):
                    row = ids[b] if ids is not None else None
                    chosen = seq_len - 1
                    if row is not None:
                        for pos in range(seq_len - 1, -1, -1):
                            if int(row[pos].item()) not in special_ids:
                                chosen = pos
                                break
                    vecs.append(h[b, chosen, :].float())
                vec = torch.stack(vecs, dim=0)
            else:
                idx = tok_idx % h.shape[1]
                vec = h[:, idx, :].float()
            captured[li].append(vec.cpu())
        return _hook

    handles: list[RemovableHandle] = []
    try:
        for li in layer_indices:
            module = get_target_module(model, li, module_name)
            handles.append(module.register_forward_hook(_make_capture_hook(li)))

        model.eval()
        loader = DataLoader(
            _DocDataset(documents),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(num_workers > 0 and device.type == "cuda"),
            collate_fn=_make_collate(tokenizer, max_length, add_special_tokens),
        )
        for inputs in loader:
            attn_mask = inputs.get("attention_mask")
            mask_holder[0] = attn_mask.to(device, non_blocking=True) if attn_mask is not None else None
            ids_holder[0] = inputs.get("input_ids")
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            model(**inputs)
    finally:
        for h in handles:
            h.remove()
        mask_holder[0] = None
        ids_holder[0] = None

    return {li: torch.cat(captured[li], dim=0) for li in layer_indices}


@torch.inference_mode()
def extract_embedding_layer(
    model: Any,
    tokenizer: Any,
    documents: list[str],
    token_position: int | str,
    batch_size: int,
    max_length: int,
    add_special_tokens: bool,
    num_workers: int = 0,
) -> torch.Tensor:
    """Extract per-document vectors directly from the input embedding layer.

    No transformer forward pass — just ``embed_tokens(input_ids)`` reduced by
    ``token_position`` (``'mean'`` masks padding, ``-1`` = last token, etc.).

    Returns:
        Tensor of shape ``[N_docs, hidden_dim]``.
    """
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_module = model.model.embed_tokens
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        embed_module = model.transformer.wte
    else:
        raise ValueError("Cannot locate embedding layer on this model architecture.")

    device = next(model.parameters()).device
    out: list[torch.Tensor] = []

    loader = DataLoader(
        _DocDataset(documents),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0 and device.type == "cuda"),
        collate_fn=_make_collate(tokenizer, max_length, add_special_tokens),
    )
    for inputs in loader:
        ids = inputs["input_ids"].to(device, non_blocking=True)
        mask = inputs.get("attention_mask")
        mask = mask.to(device, non_blocking=True) if mask is not None else None
        h = embed_module(ids).float()
        seq_len = h.shape[1]
        tok_idx = _resolve_token_index(seq_len, token_position)
        if tok_idx is None:
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                vec = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                vec = h.mean(dim=1)
        elif tok_idx == _MAX_POOL_SENTINEL:
            vec = h.max(dim=1).values
        else:
            idx = tok_idx % seq_len
            vec = h[:, idx, :]
        out.append(vec.cpu())

    return torch.cat(out, dim=0)
