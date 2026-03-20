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


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _get_system_ram_gib() -> float:
    """Return total system RAM in GiB, with fallback."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Fallback: read /proc/meminfo on Linux
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    return 32.0  # Conservative fallback


def _is_kaggle_environment() -> bool:
    """Detect if running inside Kaggle."""
    return os.path.exists("/kaggle") or os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None


def _is_t4_class_gpu(device_idx: int) -> bool:
    """Check if GPU is T4-class (under 16 GiB VRAM)."""
    if not torch.cuda.is_available():
        return False
    try:
        total_gib = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
        return total_gib < 16.5  # T4 is ~15 GiB
    except Exception:
        return False


def _parse_max_memory_override(raw: str, device_count: int) -> dict[int | str, str]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return {}

    budgets: dict[int | str, str] = {}
    if len(parts) == 1:
        budget = f"{parts[0]}GiB"
        for idx in range(device_count):
            budgets[idx] = budget
    else:
        for idx, part in enumerate(parts[:device_count]):
            budgets[idx] = f"{part}GiB"
    return budgets


def _pre_load_cleanup() -> None:
    """Clear memory before heavy model load."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _resolve_loader_kwargs(
    *,
    use_4bit: bool,
    device_map_override: Optional[str | dict],
) -> dict:
    """Resolve device placement/offload kwargs for ``from_pretrained``.

    Environment overrides:
    - ``LYRICS_DEVICE_MAP``: explicit device-map string/dict-like string
    - ``LYRICS_MAX_MEMORY_GB``: per-GPU GiB budget (single value or comma list)
    - ``LYRICS_CPU_MAX_MEMORY_GB``: CPU max memory budget (auto-detected if not set)
    - ``LYRICS_ENABLE_CPU_OFFLOAD``: enable ``offload_folder`` + ``offload_state_dict``
    - ``LYRICS_OFFLOAD_DIR``: override offload folder path

    Kaggle T4 x2 gets tighter defaults automatically:
    - GPU 0 headroom: 4.5 GiB (was 3)
    - GPU 1+ headroom: 3 GiB (was 2)
    - offload_state_dict enabled by default for multi-GPU 4-bit
    """
    if device_map_override is not None:
        return {"device_map": device_map_override, "max_memory": None}

    if not torch.cuda.is_available():
        return {"device_map": None, "max_memory": None}

    env_device_map = os.getenv("LYRICS_DEVICE_MAP")
    if env_device_map:
        dm: str | dict = env_device_map
    elif use_4bit and torch.cuda.device_count() > 1:
        dm = "balanced_low_0"
    else:
        dm = "auto"

    n_gpus = torch.cuda.device_count()
    is_kaggle = _is_kaggle_environment()
    is_multi_gpu_4bit = use_4bit and n_gpus > 1

    max_memory = None
    if is_multi_gpu_4bit:
        max_memory = {}
        override = os.getenv("LYRICS_MAX_MEMORY_GB", "").strip()
        if override:
            max_memory.update(_parse_max_memory_override(override, n_gpus))
        else:
            for i in range(n_gpus):
                total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                # Tighter headroom for T4-class GPUs (under 16 GiB)
                if _is_t4_class_gpu(i):
                    headroom_gib = 4.5 if i == 0 else 3.0
                else:
                    headroom_gib = 3.0 if i == 0 else 2.0
                budget_gib = max(8, int(total_gib - headroom_gib))
                max_memory[i] = f"{budget_gib}GiB"

        # Auto-detect CPU budget from system RAM
        cpu_override = os.getenv("LYRICS_CPU_MAX_MEMORY_GB", "").strip()
        if cpu_override:
            cpu_budget = cpu_override
        else:
            system_ram = _get_system_ram_gib()
            # Reserve ~4 GiB for system, cap at 48 GiB for safety
            cpu_budget = str(min(48, max(8, int(system_ram - 4))))
        max_memory["cpu"] = f"{cpu_budget}GiB"

    loader_kwargs = {"device_map": dm, "max_memory": max_memory}

    # Enable offload_state_dict by default for multi-GPU 4-bit on Kaggle
    # or when explicitly requested via env
    enable_offload = _env_flag("LYRICS_ENABLE_CPU_OFFLOAD") or (is_multi_gpu_4bit and is_kaggle)
    if enable_offload:
        offload_dir = os.getenv("LYRICS_OFFLOAD_DIR", "").strip()
        if not offload_dir:
            offload_dir = str(Path.cwd() / ".offload")
        loader_kwargs["offload_folder"] = offload_dir
        loader_kwargs["offload_state_dict"] = True
        # Ensure offload directory exists
        Path(offload_dir).mkdir(parents=True, exist_ok=True)

    return loader_kwargs


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

    loader_kwargs = _resolve_loader_kwargs(
        use_4bit=use_4bit,
        device_map_override=device_map_override,
    )
    dm = loader_kwargs["device_map"]
    max_memory = loader_kwargs["max_memory"]

    is_kaggle = _is_kaggle_environment()
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    t4_detected = any(_is_t4_class_gpu(i) for i in range(n_gpus)) if n_gpus > 0 else False

    print(
        "[checkpoint_loader] Placement:",
        f"device_map={dm}",
        f"max_memory={max_memory}",
        f"cpu_offload={'on' if loader_kwargs.get('offload_folder') else 'off'}",
    )
    print(
        "[checkpoint_loader] Environment:",
        f"kaggle={is_kaggle}",
        f"n_gpus={n_gpus}",
        f"t4_class={t4_detected}",
    )

    # Clean memory before heavy model load
    _pre_load_cleanup()

    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb_config,
        device_map=dm,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        **{
            key: value
            for key, value in loader_kwargs.items()
            if key not in {"device_map", "max_memory"}
        },
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
