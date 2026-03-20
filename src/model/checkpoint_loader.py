"""
Unified checkpoint loader for inference and training.

Every entry point should use these helpers instead of ad-hoc model loading.
This keeps token injection, PEFT loading, and device handling consistent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel
from peft import PeftModel

from configs.genres import SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEV_MODEL = "gpt2"
PROD_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def get_default_model() -> str:
    """Return the model path from the environment, falling back to dev."""
    return os.getenv("LYRICS_MODEL_PATH", DEV_MODEL)


# ---------------------------------------------------------------------------
# Device helpers (centralised from engine.py)
# ---------------------------------------------------------------------------

def _normalize_device(value: object) -> str:
    """Turn ``0``, ``"0"``, ``"cuda:0"`` into a consistent ``"cuda:N"``."""
    if isinstance(value, int):
        return f"cuda:{value}"
    text = str(value).strip()
    if text.isdigit():
        return f"cuda:{text}"
    return text


def _resolve_input_device(model: PreTrainedModel, fallback: str = "cpu") -> str:
    """Best-effort resolution of the device where input tensors should live."""
    # 1) model.device if it looks usable
    try:
        dev = _normalize_device(getattr(model, "device", fallback))
        if dev not in {"disk", "meta"}:
            return dev
    except Exception:
        pass
    # 2) embedding weight device
    try:
        dev = _normalize_device(model.get_input_embeddings().weight.device)
        if dev not in {"cpu", "disk", "meta"}:
            return dev
    except Exception:
        pass
    # 3) walk the hf_device_map
    hf_map = (
        getattr(model, "hf_device_map", None)
        or getattr(getattr(model, "base_model", None), "hf_device_map", None)
        or getattr(
            getattr(getattr(model, "base_model", None), "model", None),
            "hf_device_map",
            None,
        )
    )
    if hf_map:
        for key, target in hf_map.items():
            v = _normalize_device(target)
            if v.startswith("cuda"):
                return v
    return _normalize_device(fallback)


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def _load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    new_tokens = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return tokenizer


def _resolve_kaggle_loader_kwargs(
    use_4bit: bool,
    device_map_override: Optional[str | dict],
) -> dict:
    """Build kwargs for robust 4-bit loading on Kaggle multi-GPU notebooks.

    The defaults are conservative and can be overridden by environment vars:
    - ``LYRICS_DEVICE_MAP`` (string or ``auto``/``balanced_low_0``)
    - ``LYRICS_MAX_MEMORY_GB`` (per-GPU GiB budget, e.g. ``13``)
    - ``LYRICS_CPU_MAX_MEMORY_GB`` (CPU GiB budget, default ``48``)
    - ``LYRICS_ENABLE_CPU_OFFLOAD`` (``1``/``0``)
    """
    if device_map_override is not None:
        device_map = device_map_override
    elif torch.cuda.is_available() and use_4bit and torch.cuda.device_count() > 1:
        device_map = os.getenv("LYRICS_DEVICE_MAP", "balanced_low_0")
    elif torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = None

    kwargs: dict = {"device_map": device_map}
    if not (torch.cuda.is_available() and use_4bit):
        return kwargs

    max_memory_gb = os.getenv("LYRICS_MAX_MEMORY_GB")
    if max_memory_gb:
        gpu_budget = f"{max_memory_gb}GiB"
        max_memory = {i: gpu_budget for i in range(torch.cuda.device_count())}
        cpu_budget = os.getenv("LYRICS_CPU_MAX_MEMORY_GB", "48")
        max_memory["cpu"] = f"{cpu_budget}GiB"
        kwargs["max_memory"] = max_memory
    elif torch.cuda.device_count() > 1:
        # Kaggle T4 has ~15 GiB. Reserve headroom to avoid allocator spikes.
        kwargs["max_memory"] = {i: "13GiB" for i in range(torch.cuda.device_count())}
        kwargs["max_memory"]["cpu"] = os.getenv("LYRICS_CPU_MAX_MEMORY_GB", "48GiB")

    if os.getenv("LYRICS_ENABLE_CPU_OFFLOAD", "0") == "1":
        kwargs["offload_folder"] = os.getenv("LYRICS_OFFLOAD_DIR", "/tmp/lyrics_offload")
        kwargs["offload_state_dict"] = True

    return kwargs


def load_base(
    model_name: Optional[str] = None,
    use_4bit: bool = False,
    device_map_override: Optional[str | dict] = None,
) -> tuple[PreTrainedModel, AutoTokenizer]:
    """Load the raw base model + tokenizer with special tokens injected.

    This is the single place that calls ``from_pretrained`` for training
    **and** inference.  Both :func:`load_for_inference` and
    :func:`load_for_training` delegate here.
    """
    name = model_name or get_default_model()
    print(f"[checkpoint_loader] Loading base model: {name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    loader_kwargs = _resolve_kaggle_loader_kwargs(
        use_4bit=use_4bit,
        device_map_override=device_map_override,
    )
    print(
        "[checkpoint_loader] Loader placement:",
        {
            "device_map": loader_kwargs.get("device_map"),
            "max_memory": loader_kwargs.get("max_memory"),
            "cpu_offload": bool(loader_kwargs.get("offload_folder")),
        },
    )

    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        **loader_kwargs,
    )

    tokenizer = _load_tokenizer(name)
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    return model, tokenizer


def _load_peft_adapter(model: PreTrainedModel, checkpoint_path: str) -> PreTrainedModel:
    """Load PEFT weights in a Kaggle-safe way.

    We prefer CPU-first adapter loading so small GPUs do not spike during
    safetensor materialization. Older PEFT versions may not support every
    keyword, so we fall back conservatively.
    """
    peft_kwargs = {
        "is_trainable": False,
        "low_cpu_mem_usage": True,
        "torch_device": "cpu",
    }
    if torch.cuda.is_available():
        peft_kwargs["ephemeral_gpu_offload"] = True

    try:
        return PeftModel.from_pretrained(model, checkpoint_path, **peft_kwargs)
    except TypeError:
        peft_kwargs.pop("ephemeral_gpu_offload", None)
        return PeftModel.from_pretrained(model, checkpoint_path, **peft_kwargs)


# ---------------------------------------------------------------------------
# Inference loader
# ---------------------------------------------------------------------------

def load_for_inference(
    checkpoint_path: Optional[str] = None,
    base_model: Optional[str] = None,
    use_4bit: bool = True,
    device_map_override: Optional[str | dict] = None,
) -> tuple[PreTrainedModel, AutoTokenizer, str]:
    """Load a model ready for inference.

    Parameters
    ----------
    checkpoint_path:
        Path to a PEFT / LoRA checkpoint directory.  If ``None``, the raw
        base model is returned (useful for smoke tests with GPT-2).
    base_model:
        HuggingFace model id.  Defaults to ``LYRICS_MODEL_PATH`` env var
        or ``gpt2`` for local dev.
    use_4bit:
        Use 4-bit quantisation when a CUDA device is available.
    device_map_override:
        Force a specific device-map (e.g. ``{"": 0}`` for single GPU on
        Kaggle).

    Returns
    -------
    (model, tokenizer, device)
        Ready-to-generate model, tokenizer, and the device string for
        input tensors.
    """
    model, tokenizer = load_base(
        model_name=base_model,
        use_4bit=use_4bit,
        device_map_override=device_map_override,
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[checkpoint_loader] Loading PEFT adapter: {checkpoint_path}")
        model = _load_peft_adapter(model, checkpoint_path)
        model.eval()
        # Keep adapters attached for 4-bit inference. Merging can materialize a
        # full-precision model on CPU and cause NaN/Inf sampling in constrained
        # Kaggle environments.
        if use_4bit and torch.cuda.is_available():
            print("[checkpoint_loader] Keeping LoRA adapter attached for 4-bit GPU inference")
        else:
            try:
                model = model.merge_and_unload()
                print("[checkpoint_loader] LoRA weights merged for fast inference")
            except Exception:
                print("[checkpoint_loader] Running with LoRA adapter (merge not available)")

    model.eval()
    device = _resolve_input_device(model)
    if device == "cpu":
        try:
            model = model.float()
            print("[checkpoint_loader] Promoted CPU inference model to float32")
        except Exception:
            pass
    print(f"[checkpoint_loader] Ready for inference on {device}")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Training loader  (thin wrapper — SFT still owns the LoRA application)
# ---------------------------------------------------------------------------

def load_for_training(
    base_model: Optional[str] = None,
    use_4bit: bool = True,
) -> tuple[PreTrainedModel, AutoTokenizer]:
    """Load the base model for fine-tuning.

    LoRA application, gradient checkpointing, and accelerator setup are
    handled by the caller (``sft.py``).
    """
    return load_base(
        model_name=base_model,
        use_4bit=use_4bit,
        device_map_override=None,
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing checkpoint_loader with GPT-2 …")
    mdl, tok, dev = load_for_inference(base_model="gpt2", use_4bit=False)
    ids = tok("Hello world", return_tensors="pt").to(dev)
    out = mdl.generate(**ids, max_new_tokens=10)
    print(f"  Generated: {tok.decode(out[0], skip_special_tokens=True)}")
    print("  OK")
