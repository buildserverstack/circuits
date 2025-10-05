"""Register forward/backward hooks for modules listed; store {'pre', 'post', 'grad'} tensors by module path. Avoid retaining graph unnecessarily. Use weakrefs where appropriate."""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn

TensorDict = Dict[str, torch.Tensor]


@dataclass
class RunCache:
    """Container for cached tensors keyed by module path."""

    storage: Dict[str, TensorDict] = field(default_factory=dict)

    def add_tensor(self, module_path: str, kind: str, tensor: torch.Tensor) -> None:
        bucket = self.storage.setdefault(module_path, {})
        bucket[kind] = tensor.detach().cpu()

    def get(self, module_path: str) -> TensorDict:
        return self.storage.get(module_path, {})

    def clear(self) -> None:
        self.storage.clear()


HookHandle = torch.utils.hooks.RemovableHandle
HookFactory = Callable[[nn.Module, str], HookHandle]


def _module_path(module: nn.Module, prefix: Optional[str] = None) -> str:
    name = getattr(module, "_nnviz_name", None)
    if name is not None:
        return name
    return prefix or module.__class__.__name__


def register_module_names(root: nn.Module) -> None:
    """Annotate modules with dotted paths for consistent hook keys."""

    for name, module in root.named_modules():
        module._nnviz_name = name or root.__class__.__name__  # type: ignore[attr-defined]


def register_hooks(
    model: nn.Module,
    modules: Iterable[nn.Module],
    cache: RunCache,
    with_backward: bool = True,
) -> Dict[str, HookHandle]:
    """Register hooks on specified modules and populate the cache."""

    handles: Dict[str, HookHandle] = {}

    def forward_pre_hook(module: nn.Module, inputs: tuple) -> None:
        module_path = _module_path(module)
        if inputs:
            cache.add_tensor(module_path, "pre", inputs[0])

    def forward_hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        module_path = _module_path(module)
        cache.add_tensor(module_path, "post", output)

    def backward_hook(module: nn.Module, grad_input, grad_output) -> None:  # type: ignore[override]
        module_path = _module_path(module)
        if grad_output:
            cache.add_tensor(module_path, "grad", grad_output[0])

    for module in modules:
        module_path = _module_path(module)
        handles[f"{module_path}.pre"] = module.register_forward_pre_hook(forward_pre_hook)
        handles[f"{module_path}.post"] = module.register_forward_hook(forward_hook)
        if with_backward:
            handles[f"{module_path}.back"] = module.register_full_backward_hook(backward_hook)
    return handles


def unregister_hooks(handles: Dict[str, HookHandle]) -> None:
    """Remove registered hooks safely."""

    for handle in handles.values():
        with contextlib.suppress(RuntimeError):
            handle.remove()


def ensure_run_cache(cache: Optional[RunCache] = None) -> RunCache:
    return cache if cache is not None else RunCache()


def register_attention_hook(module: nn.Module, cache: RunCache, name: Optional[str] = None) -> HookHandle:
    """Register a hook capturing attention weights."""

    def hook(_, __, output):
        module_path = name or _module_path(module)
        cache.add_tensor(module_path, "attention", output)

    return module.register_forward_hook(hook)


__all__ = [
    "RunCache",
    "register_module_names",
    "register_hooks",
    "unregister_hooks",
    "ensure_run_cache",
    "register_attention_hook",
]
