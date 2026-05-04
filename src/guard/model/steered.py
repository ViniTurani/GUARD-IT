"""SteeredModel — HuggingFace-compatible wrapper for benchmark integration."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from torch.utils.hooks import RemovableHandle

from guard.config.normalization import NormalizationConfig
from guard.online.hook import register_steering_hook
from guard.storage.io import load_sv

if TYPE_CHECKING:
    from guard.config.gate import GateConfig
    from guard.online.gate import SimilarityGate

__all__ = ["SteeredModel"]


class SteeredModel:
    """HuggingFace-compatible model wrapper that applies one or more SVs at inference.

    ``SteeredModel`` is designed to be a **drop-in replacement** for any
    ``AutoModelForCausalLM`` instance in benchmark evaluation loops (TOFU,
    MUSE, etc.).  It forwards all attribute lookups, ``__call__``, and
    ``generate`` to the wrapped model after registering the steering hook.

    Key design decisions:

    * **Single ``coeff_holder``** — one mutable ``list[float]`` shared by all
      hooks.  Updating ``steered.coeff`` changes the coefficient for every
      layer in a single assignment, with no hook re-registration.
    * **Context manager** — ``__exit__`` removes all hooks, restoring the
      base model to its original behaviour.
    * **``__getattr__`` delegation** — attributes not defined on this wrapper
      fall through to the underlying model, making it transparent to evaluators.

    Construction is done via class methods to make intent explicit:

    * :meth:`from_sv_path` — single SV from a ``sv.pt`` file.
    * :meth:`from_sv_dir` — infer the path from standard directory layout.
    * :meth:`from_cluster_dir` — gated (multi-cluster) routing via
      :class:`~guard.gateway.router.SimilarityGate`.

    Example::

        with SteeredModel.from_sv_path(
            model, sv_path="steering_vectors/.../sv.pt",
            coeff=0.0, layer_idx=8, rotation_only=True,
        ) as steered:
            for coeff in [0.8, 0.3, 0.15, 0.0]:
                steered.coeff = coeff
                result = evaluator.evaluate(model=steered, tokenizer=tokenizer, ...)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: Any,
        coeff_holder: list[float],
        handles: list[RemovableHandle],
        gate: "SimilarityGate | None" = None,
        cluster_layer: int | str | None = None,
    ) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_coeff_holder", coeff_holder)
        object.__setattr__(self, "_handles", handles)
        object.__setattr__(self, "_gate", gate)
        object.__setattr__(self, "_cluster_layer", cluster_layer)

    @classmethod
    def from_sv_path(
        cls,
        model: Any,
        sv_path: Path | str,
        coeff: float,
        layer_idx: int,
        module_name: str = "residual",
        rotation_only: bool = True,
    ) -> "SteeredModel":
        """Create a steered model from a single ``sv.pt`` file.

        Args:
            model: Base HuggingFace causal LM.
            sv_path: Path to the ``sv.pt`` file.
            coeff: Initial steering coefficient α.
            layer_idx: Transformer layer to steer.
            module_name: Module to hook (``'residual'``, ``'mlp'``, etc.).
            rotation_only: If ``True`` (default), apply rotation-only steering (Eq. 8).

        Returns:
            A :class:`SteeredModel` with one hook registered.
        """
        sv = load_sv(sv_path)
        coeff_holder: list[float] = [float(coeff)]
        handle = register_steering_hook(
            model, sv, coeff_holder, layer_idx, module_name, rotation_only
        )
        logger.info(
            "SteeredModel: layer={}  module={}  coeff={:.4f}  rotation_only={}",
            layer_idx, module_name, coeff, rotation_only,
        )
        return cls(model, coeff_holder, [handle])

    @classmethod
    def from_sv_dir(
        cls,
        model: Any,
        sv_dir: Path | str,
        coeff: float,
        layer_indices: list[int] | None = None,
        module_name: str = "residual",
        rotation_only: bool = True,
    ) -> "SteeredModel":
        """Create a steered model from a layer directory, hooking one or more layers.

        Expects ``sv.pt`` files inside ``{sv_dir}/{module_name}/layer_{idx}/sv.pt``.
        If ``layer_indices`` is ``None``, all ``layer_*`` subdirectories found
        under ``sv_dir/{module_name}/`` are loaded.

        Args:
            model: Base HuggingFace causal LM.
            sv_dir: Directory containing the ``{module_name}/layer_*/sv.pt`` layout.
            coeff: Initial steering coefficient α (shared across all layers).
            layer_indices: Explicit list of layer indices to load, or ``None``
                to auto-discover all available layers.
            module_name: Module to hook.
            rotation_only: Rotation-only steering flag.

        Returns:
            A :class:`SteeredModel` with one hook per loaded layer.
        """
        sv_dir = Path(sv_dir)
        coeff_holder: list[float] = [float(coeff)]
        handles: list[RemovableHandle] = []

        if layer_indices is None:
            layer_dirs = sorted(
                (sv_dir / module_name).glob("layer_*"),
                key=lambda p: int(p.name.split("_")[1]),
            )
            layer_indices = [int(p.name.split("_")[1]) for p in layer_dirs]

        for li in layer_indices:
            pt = sv_dir / module_name / f"layer_{li}" / "sv.pt"
            sv = load_sv(pt)
            handle = register_steering_hook(
                model, sv, coeff_holder, li, module_name, rotation_only
            )
            handles.append(handle)
            logger.info("SteeredModel: loaded layer={} from {}", li, pt)

        return cls(model, coeff_holder, handles)

    @classmethod
    def from_normalization_config(
        cls,
        model: Any,
        sv_path: Path | str,
        coeff: float,
        layer_idx: int,
        norm_cfg: NormalizationConfig,
        module_name: str = "residual",
    ) -> "SteeredModel":
        """Convenience factory that reads ``rotation_only`` from a :class:`NormalizationConfig`.

        Args:
            model: Base HuggingFace causal LM.
            sv_path: Path to ``sv.pt``.
            coeff: Initial steering coefficient α.
            layer_idx: Transformer layer index.
            norm_cfg: :class:`NormalizationConfig` instance.
            module_name: Module to hook.

        Returns:
            A :class:`SteeredModel`.
        """
        return cls.from_sv_path(
            model,
            sv_path,
            coeff=coeff,
            layer_idx=layer_idx,
            module_name=module_name,
            rotation_only=norm_cfg.rotation_only,
        )

    @classmethod
    def from_cluster_dir(
        cls,
        model: Any,
        cluster_dir: Path | str,
        coeff: float,
        gate_cfg: "GateConfig",
        layer_idx: int,
        module_name: str = "residual",
        tokenizer: Any | None = None,
    ) -> "SteeredModel":
        """Create a gated steered model from a cluster SV directory.

        Loads ``centroids.pt``, ``psv_clusters.pt``, and (for text routing)
        ``text_centroids.pt`` from ``cluster_dir``.  Registers a single
        :class:`~guard.gateway.router.SimilarityGate` hook on ``layer_idx``.

        For backward compatibility, falls back to loading ``sv_clusters.pt``
        if ``psv_clusters.pt`` is not found (legacy experiment directories).

        Args:
            model: Base HuggingFace causal LM.
            cluster_dir: Directory produced by the clustering pipeline,
                containing ``centroids.pt``, ``psv_clusters.pt``, and optionally
                ``text_centroids.pt``.
            coeff: Initial steering coefficient α.
            gate_cfg: :class:`~guard.config.GateConfig` instance.
            layer_idx: Transformer layer to steer.
            module_name: Module to hook.
            tokenizer: Required when ``gate_cfg.routing_source == 'text'``.

        Returns:
            A :class:`SteeredModel` with a :class:`~guard.gateway.router.SimilarityGate`
            attached.
        """
        from guard.online.gate import SimilarityGate

        cluster_dir = Path(cluster_dir)
        coeff_holder: list[float] = [float(coeff)]

        behavior_dir = cluster_dir.parent.parent
        meta_path = behavior_dir / "cluster_meta.json"
        rotation_only: bool = True
        cluster_layer: int | str | None = None
        if meta_path.exists():
            _meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rotation_only = bool(
                _meta.get("rotation_only", _meta.get("norm_activation_coeff", True))
            )
            cluster_layer = _meta.get("cluster_layer")

        centroids: torch.Tensor = torch.load(
            cluster_dir / "centroids.pt", map_location="cpu", weights_only=True
        )
        psv_clusters = _load_psv_clusters(cluster_dir)

        text_centroids_path = cluster_dir / "text_centroids.pt"
        text_centroids: torch.Tensor | None = None
        if text_centroids_path.exists():
            text_centroids = torch.load(text_centroids_path, map_location="cpu", weights_only=True)

        cluster_texts: list[list[str]] | None = None
        if gate_cfg.routing_mode == "bm25":
            behavior_dir = cluster_dir.parent.parent
            labels_path = behavior_dir / "labels.pt"
            docs_path = behavior_dir / "forget_docs.json"
            if labels_path.exists() and docs_path.exists():
                labels = torch.load(labels_path, map_location="cpu", weights_only=True)
                all_docs = json.loads(docs_path.read_text(encoding="utf-8"))
                k = centroids.shape[0]
                cluster_texts = [[] for _ in range(k)]
                for doc, lbl in zip(all_docs, labels.tolist()):
                    if 0 <= int(lbl) < k:
                        cluster_texts[int(lbl)].append(doc)

        retain_text_centroid: torch.Tensor | None = None
        if gate_cfg.routing_mode == "ratio" or gate_cfg.retain_threshold > 0:
            behavior_dir = cluster_dir.parent.parent
            retain_emb_path = behavior_dir / "retain_embeddings.pt"
            if retain_emb_path.exists():
                retain_emb = torch.load(
                    retain_emb_path, map_location="cpu", weights_only=True
                ).float()
                import torch.nn.functional as _F
                retain_text_centroid = _F.normalize(retain_emb.mean(dim=0), dim=-1)

        llama_centroids: torch.Tensor | None = None
        if gate_cfg.routing_source == "llama_mean":
            llama_path = cluster_dir / "llama_centroids.pt"
            if llama_path.exists():
                llama_centroids = torch.load(llama_path, map_location="cpu", weights_only=True)

        gate = SimilarityGate(
            centroids=centroids,
            psv_clusters=psv_clusters,
            gate_cfg=gate_cfg,
            text_centroids=text_centroids,
            tokenizer=tokenizer,
            cluster_texts=cluster_texts,
            retain_text_centroid=retain_text_centroid,
            llama_centroids=llama_centroids,
            rotation_only=rotation_only,
        )

        handles: list[RemovableHandle] = []
        sv_handle = gate.register_hook(
            model=model,
            coeff_holder=coeff_holder,
            layer_idx=layer_idx,
            module_name=module_name,
        )
        handles.append(sv_handle)

        if (
            gate_cfg.routing_source in ("activation", "activation_euclidean", "llama_mean")
            and cluster_layer is not None
            and cluster_layer != layer_idx
        ):
            if isinstance(cluster_layer, int) and cluster_layer > layer_idx:
                raise ValueError(
                    f"cluster_layer={cluster_layer} is deeper than SV layer={layer_idx}. "
                    "Routing would read stale scores from the previous forward pass. "
                    f"Either set SV layers >= {cluster_layer}, or re-cluster with "
                    f"cluster_layer <= {layer_idx}."
                )
            scoring_handle = gate.register_scoring_hook(
                model=model, cluster_layer=cluster_layer, module_name=module_name
            )
            if isinstance(scoring_handle, list):
                handles.extend(scoring_handle)
            else:
                handles.append(scoring_handle)
            logger.info(
                "SteeredModel (gate): SV@layer={}  scoring@{}  routing={}  threshold={:.4f}",
                layer_idx, cluster_layer, gate_cfg.routing_source, gate_cfg.threshold,
            )
        else:
            logger.info(
                "SteeredModel (gate): layer={}  routing={}  threshold={:.4f}",
                layer_idx, gate_cfg.routing_source, gate_cfg.threshold,
            )

        if gate_cfg.routing_source in ("activation", "activation_euclidean", "llama_mean"):
            mask_handle = gate.register_mask_capture(model)
            handles.append(mask_handle)

        return cls(model, coeff_holder, handles, gate=gate, cluster_layer=cluster_layer)

    # ------------------------------------------------------------------
    # Coefficient property
    # ------------------------------------------------------------------

    @property
    def coeff(self) -> float:
        """Current steering coefficient α (shared across all registered layers)."""
        return float(object.__getattribute__(self, "_coeff_holder")[0])

    @coeff.setter
    def coeff(self, value: float) -> None:
        object.__getattribute__(self, "_coeff_holder")[0] = float(value)

    def update_coeff(self, value: float) -> None:
        """Update the steering coefficient (alias for ``steered.coeff = value``)."""
        self.coeff = value

    # ------------------------------------------------------------------
    # Hook lifecycle
    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Remove all registered hooks, restoring the base model."""
        for h in object.__getattribute__(self, "_handles"):
            h.remove()
        object.__getattribute__(self, "_handles").clear()
        logger.debug("SteeredModel: all hooks removed.")

    def __enter__(self) -> "SteeredModel":
        return self

    def __exit__(self, *_: Any) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # HuggingFace-compatible interface
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device of the underlying model."""
        model = object.__getattribute__(self, "_model")
        device: torch.device = next(model.parameters()).device
        return device

    @property
    def config(self) -> Any:
        """Model config (pass-through to the underlying model)."""
        return object.__getattribute__(self, "_model").config

    def __call__(self, input_ids: Any = None, labels: Any = None, **kwargs: Any) -> Any:
        gate = object.__getattribute__(self, "_gate")
        model = object.__getattribute__(self, "_model")
        cluster_layer = object.__getattribute__(self, "_cluster_layer")
        if gate is not None and input_ids is not None:
            gate.cache_routing(input_ids)
            if cluster_layer == "mean_all":
                gate.cache_mean_all_routing(model, input_ids, kwargs.get("attention_mask"))
        return model(input_ids=input_ids, labels=labels, **kwargs)

    def generate(self, input_ids: Any = None, **kwargs: Any) -> Any:
        """Generate text with steering applied.

        For text-routing gate mode, the sentence-transformer embedding is
        computed once from the initial prompt and cached for all autoregressive
        steps.  For ``cluster_layer='mean_all'`` routing, a separate no-steering
        forward pass collects activations from all transformer layers to
        compute the global mean.
        """
        gate = object.__getattribute__(self, "_gate")
        model = object.__getattribute__(self, "_model")
        cluster_layer = object.__getattribute__(self, "_cluster_layer")
        if gate is not None and input_ids is not None:
            gate.cache_routing(input_ids)
            if cluster_layer == "mean_all":
                gate.cache_mean_all_routing(model, input_ids, kwargs.get("attention_mask"))
        result = model.generate(input_ids=input_ids, **kwargs)
        if gate is not None:
            gate.clear_cache()
        return result

    def __getattr__(self, name: str) -> Any:
        try:
            model = object.__getattribute__(self, "_model")
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def __repr__(self) -> str:
        model = object.__getattribute__(self, "_model")
        coeff = object.__getattribute__(self, "_coeff_holder")[0]
        n_hooks = len(object.__getattribute__(self, "_handles"))
        return f"SteeredModel(model={type(model).__name__}, coeff={coeff}, n_hooks={n_hooks})"


def _load_psv_clusters(cluster_dir: Path) -> torch.Tensor:
    """Load ``psv_clusters.pt``, falling back to legacy ``sv_clusters.pt``."""
    psv_path = cluster_dir / "psv_clusters.pt"
    if psv_path.exists():
        return torch.load(psv_path, map_location="cpu", weights_only=True)

    legacy_path = cluster_dir / "sv_clusters.pt"
    if legacy_path.exists():
        logger.warning(
            "Found legacy 'sv_clusters.pt' in {} — rename it to 'psv_clusters.pt': "
            "mv '{}' '{}'",
            cluster_dir, legacy_path, psv_path,
        )
        return torch.load(legacy_path, map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        f"Neither 'psv_clusters.pt' nor 'sv_clusters.pt' found in {cluster_dir}."
    )
