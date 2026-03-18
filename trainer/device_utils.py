import contextlib
import os
import torch


def _normalize_backend(name: str) -> str:
    if not name:
        return ""
    name = name.strip().lower()
    if name in {"gpu", "cuda"}:
        return "cuda"
    return name


def _module_is_available(name: str) -> bool:
    mod = getattr(torch, name, None)
    if mod is None:
        return False
    is_available = getattr(mod, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    return False


def detect_backend() -> str:
    env_backend = _normalize_backend(os.getenv("ABW_BACKEND") or os.getenv("TORCH_DEVICE") or "")
    if env_backend:
        return env_backend
    if _module_is_available("cuda"):
        return "cuda"
    if _module_is_available("xpu"):
        return "xpu"
    if _module_is_available("npu"):
        return "npu"
    if _module_is_available("mlu"):
        return "mlu"
    if _module_is_available("musa"):
        return "musa"
    return "cpu"


def get_device_module(backend: str):
    backend = _normalize_backend(backend)
    if backend in {"", "cpu"}:
        return None
    return getattr(torch, backend, None)


def device_count(backend: str | None = None) -> int:
    backend = _normalize_backend(backend or detect_backend())
    mod = get_device_module(backend)
    if mod is not None and hasattr(mod, "device_count"):
        try:
            return int(mod.device_count())
        except Exception:
            return 0
    return 1 if backend == "cpu" else 0


def pl_accelerator(backend: str) -> str:
    override = os.getenv("ABW_ACCELERATOR") or ""
    if override:
        return override
    backend = _normalize_backend(backend)
    if backend == "cpu":
        return "cpu"
    if backend == "cuda":
        return "gpu"
    # For vendor backends with Lightning plugins, pass through.
    return backend


def autocast_context(backend: str, enabled: bool = True):
    if not enabled:
        return contextlib.nullcontext()
    backend = _normalize_backend(backend)
    try:
        return torch.autocast(device_type=backend, enabled=enabled)
    except Exception:
        if backend == "cuda" and hasattr(torch.cuda, "amp"):
            return torch.cuda.amp.autocast(enabled=enabled)
        return contextlib.nullcontext()
