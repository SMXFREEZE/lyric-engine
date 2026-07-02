"""Pytest configuration.

- Makes the repo root importable as `src.*` / `configs.*`.
- Stubs `torch` / `transformers` when the real stack is missing or broken so
  modules that import them at the top level (e.g. src.inference.engine) can
  still have their pure-Python parts tested. CI installs no torch at all.
"""

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _model_stack_works() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import PreTrainedModel  # noqa: F401
        return True
    except Exception:
        return False


MODEL_STACK_AVAILABLE = _model_stack_works()

if not MODEL_STACK_AVAILABLE:
    class _Stub:  # generic base class stand-in
        pass

    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = lambda: (lambda fn: fn)  # decorator form used in engine

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.PreTrainedModel = _Stub
    transformers_stub.PreTrainedTokenizer = _Stub
    transformers_stub.LogitsProcessor = _Stub
    transformers_stub.LogitsProcessorList = list

    sys.modules["torch"] = torch_stub
    sys.modules["transformers"] = transformers_stub
