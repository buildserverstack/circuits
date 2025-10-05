"""Hook helpers for capturing activations and gradients."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn


@dataclass
class _HookHandle:
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class ActivationStore:
    """Capture forward activations via hooks."""

    def __init__(self, store_pre: bool = True, store_post: bool = True, max_batches: int = 4) -> None:
        self.store_pre = store_pre
        self.store_post = store_post
        self.max_batches = max_batches
        self._storage: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        self._handles: Dict[str, _HookHandle] = {}
        self._model: nn.Module | None = None

    def attach(self, model: nn.Module, layers: List[str] | None = None) -> None:
        """Attach forward hooks to selected layers."""
        self.detach()
        self._model = model
        modules = dict(model.named_modules())
        target_layers = layers or [name for name in modules.keys() if name]
        for layer_name in target_layers:
            module = modules.get(layer_name)
            if module is None:
                continue
            self._storage.setdefault(layer_name, {"pre": [], "post": [], "inputs": [], "outputs": []})
            handle = _HookHandle()

            if self.store_pre:

                def _pre_hook(mod, inputs, name=layer_name):
                    if self._should_skip(name, "pre"):
                        return
                    storage = self._storage[name]
                    tensor = inputs[0] if inputs else None
                    if tensor is not None and isinstance(tensor, torch.Tensor):
                        storage["pre"].append(tensor.detach().cpu())
                    storage["inputs"].append(tuple(inp.detach().cpu() if isinstance(inp, torch.Tensor) else inp for inp in inputs))

                handle.handles.append(module.register_forward_pre_hook(_pre_hook, with_kwargs=False))

            if self.store_post:

                def _post_hook(mod, inputs, output, name=layer_name):
                    if self._should_skip(name, "post"):
                        return
                    storage = self._storage[name]
                    if isinstance(output, torch.Tensor):
                        storage["post"].append(output.detach().cpu())
                        storage["outputs"].append(output.detach().cpu())
                    elif isinstance(output, (tuple, list)) and output:
                        storage["post"].append(output[0].detach().cpu())
                        storage["outputs"].append(tuple(out.detach().cpu() if isinstance(out, torch.Tensor) else out for out in output))
                    else:
                        storage["post"].append(torch.tensor([]))
                        storage["outputs"].append(output)

                handle.handles.append(module.register_forward_hook(_post_hook, with_kwargs=False))

            self._handles[layer_name] = handle

    def _should_skip(self, layer_name: str, kind: str) -> bool:
        max_batches = self.max_batches
        if max_batches is None:
            return False
        storage = self._storage.get(layer_name, {})
        values = storage.get(kind, [])
        return len(values) >= max_batches

    def detach(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        self._model = None

    def get(self, layer_name: str, kind: str = "post") -> List[torch.Tensor]:
        return self._storage.get(layer_name, {}).get(kind, [])

    def clear(self) -> None:
        self._storage.clear()


class GradientStore:
    """Capture backward gradients via hooks."""

    def __init__(self, retain_graph: bool = False) -> None:
        self.retain_graph = retain_graph
        self._storage: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        self._handles: Dict[str, _HookHandle] = {}
        self._model: nn.Module | None = None

    def attach(self, model: nn.Module, layers: List[str] | None = None) -> None:
        self.detach()
        self._model = model
        modules = dict(model.named_modules())
        target_layers = layers or [name for name in modules.keys() if name]
        for layer_name in target_layers:
            module = modules.get(layer_name)
            if module is None:
                continue
            self._storage.setdefault(layer_name, {"in_grad": [], "out_grad": [], "param_grad": []})
            handle = _HookHandle()

            def _full_hook(mod, grad_input, grad_output, name=layer_name):
                storage = self._storage[name]
                if grad_input:
                    storage["in_grad"].append(tuple(g.detach().cpu() if isinstance(g, torch.Tensor) else g for g in grad_input))
                if grad_output:
                    storage["out_grad"].append(tuple(g.detach().cpu() if isinstance(g, torch.Tensor) else g for g in grad_output))

            handle.handles.append(module.register_full_backward_hook(_full_hook))

            for pname, param in module.named_parameters(recurse=False):
                def _grad_hook(grad, name=layer_name, p_name=pname):
                    storage = self._storage[name]
                    storage["param_grad"].append((p_name, grad.detach().cpu()))

                handle.handles.append(param.register_hook(_grad_hook))

            self._handles[layer_name] = handle

    def detach(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()
        self._model = None

    def get(self, layer_name: str, kind: str = "out_grad") -> List[torch.Tensor]:
        return self._storage.get(layer_name, {}).get(kind, [])

    def clear(self) -> None:
        self._storage.clear()


__all__ = ["ActivationStore", "GradientStore"]
