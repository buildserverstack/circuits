"""High-level analysis routines."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from .activations import collect_activation_batches, summarize_store
from .data import get_sample_image
from .hooks import ActivationStore, GradientStore
from .gradients import summarize_gradients
from .models import enumerate_layers, load_model
from .utils import ensure_out_dir, save_json, tensor_to_pil
from .visualize.distributions import plot_activation_hist
from .visualize.feature_maps import plot_feature_maps
from .visualize.grad_cam import grad_cam
from .visualize.graph import draw_network_graph
from .visualize.neuron_grid import plot_neuron_grid
from .visualize.saliency import saliency_map


def _select_layers(model) -> List[str]:
    selected: List[str] = []
    for name, module in enumerate_layers(model):
        selected.append(name)
        if len(selected) >= 6:
            break
    return selected


def analyze_single_image(
    model_name: str = "resnet18",
    image_path: str | None = None,
    target_class: int | None = None,
    out_dir: str = "outputs",
    layers: List[str] | None = None,
) -> Dict:
    out_dir_path = ensure_out_dir(out_dir)
    model = load_model(model_name, pretrained=True)
    layer_names = layers or _select_layers(model)

    image = get_sample_image(image_path).unsqueeze(0)
    image = image.clone().detach().requires_grad_(True)

    activation_store = ActivationStore(store_pre=True, store_post=True, max_batches=1)
    activation_store.attach(model, layer_names)

    gradient_store = GradientStore()
    gradient_store.attach(model, layer_names)

    output = model(image)
    probs = F.softmax(output, dim=1)
    pred_prob, pred_idx = probs.max(dim=1)
    if target_class is None:
        target_class = int(pred_idx.item())
    score = output[:, target_class]
    model.zero_grad()
    score.backward(torch.ones_like(score))

    post_summary = summarize_store(activation_store, layer_names, kind="post")
    pre_summary = summarize_store(activation_store, layer_names, kind="pre")
    grad_summary = summarize_gradients(gradient_store, layer_names)

    artifacts: Dict[str, object] = {}
    artifacts.setdefault("feature_maps", [])
    network_path = Path(out_dir_path) / "network_graph.png"
    artifacts["network_graph"] = draw_network_graph(model, str(network_path))

    feature_tensor = None
    for name in layer_names:
        tensors = activation_store.get(name, kind="post")
        if tensors and isinstance(tensors[0], torch.Tensor) and tensors[0].dim() == 4:
            feature_tensor = tensors[0]
            feature_path = Path(out_dir_path) / f"feature_maps_{name.replace('.', '_')}.png"
            artifacts.setdefault("feature_maps", []).append(
                plot_feature_maps(feature_tensor, save_path=str(feature_path))
            )
            break

    hist_paths: List[str] = []
    for name in layer_names:
        tensor = collect_activation_batches(activation_store, name, kind="post")
        if tensor is None:
            continue
        hist_path = Path(out_dir_path) / f"activation_hist_{name.replace('.', '_')}.png"
        hist_paths.append(plot_activation_hist(name, tensor, str(hist_path)))
    artifacts["activation_hists"] = hist_paths

    saliency_path = Path(out_dir_path) / "saliency.png"
    saliency_map(model, image, target_class=target_class, save_path=str(saliency_path))
    artifacts["saliency"] = str(saliency_path)

    grad_layer = None
    for name, module in model.named_modules():
        if hasattr(module, "weight") and isinstance(getattr(module, "weight"), torch.Tensor) and module.weight.dim() == 4:
            grad_layer = name
            break
    if grad_layer is None:
        grad_layer = layer_names[0]
    grad_cam_path = Path(out_dir_path) / f"grad_cam_{grad_layer.replace('.', '_')}.png"
    grad_cam(model, image, target_class=target_class, layer_name=grad_layer, save_path=str(grad_cam_path))
    artifacts["grad_cam"] = str(grad_cam_path)

    neuron_grid_path = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.detach()
            bias = module.bias.detach() if module.bias is not None else None
            neuron_grid_path = Path(out_dir_path) / f"neuron_grid_{name.replace('.', '_')}.png"
            artifacts["neuron_grid_fc"] = plot_neuron_grid(weight, bias, str(neuron_grid_path))
            break
    if neuron_grid_path is None:
        artifacts["neuron_grid_fc"] = None

    summary = {
        "model": model_name,
        "prediction": {
            "class_idx": int(pred_idx.item()),
            "prob": float(pred_prob.item()),
            "label": str(target_class),
        },
        "artifacts": artifacts,
        "metrics": {
            "layer_sparsity": {name: stats["sparsity"] for name, stats in post_summary.items()},
            "mean_pre_act": {name: stats.get("mean", 0.0) for name, stats in pre_summary.items()},
            "mean_post_act": {name: stats.get("mean", 0.0) for name, stats in post_summary.items()},
            "gradient_magnitude": {name: stats.get("mean_abs_grad", 0.0) for name, stats in grad_summary.items()},
        },
    }

    summary_path = Path(out_dir_path) / "summary.json"
    save_json(summary, summary_path)

    activation_store.detach()
    gradient_store.detach()

    return summary


__all__ = ["analyze_single_image"]
