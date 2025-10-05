"""Command line interface for NNViz."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


from .activations import summarize_store
from .analysis import analyze_single_image
from .data import get_cifar10
from .hooks import ActivationStore
from .models import enumerate_layers, load_model
from .utils import ensure_out_dir


def _dataset_scan(model_name: str, layers: List[str] | None, limit: int, out_dir: str) -> Dict[str, float]:
    model = load_model(model_name, pretrained=True)
    layer_names = layers or [name for name, _ in enumerate_layers(model)[:6]]
    activation_store = ActivationStore(store_pre=False, store_post=True, max_batches=1)
    activation_store.attach(model, layer_names)
    loader = get_cifar10(batch_size=1, train=False)
    accum = defaultdict(list)
    for idx, (images, _) in enumerate(loader):
        if idx >= limit:
            break
        activation_store.clear()
        with torch.no_grad():
            model(images)
        summary = summarize_store(activation_store, layer_names, kind="post")
        for name, stats in summary.items():
            accum[name].append(stats["sparsity"])
    activation_store.detach()
    aggregated = {name: float(sum(vals) / len(vals)) for name, vals in accum.items() if vals}
    ensure_out_dir(out_dir)
    out_path = Path(out_dir) / "dataset_scan.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="NNViz CLI")
    sub = parser.add_subparsers(dest="cmd")

    analyze_p = sub.add_parser("analyze", help="Analyze a single image")
    analyze_p.add_argument("--model", default="resnet18")
    analyze_p.add_argument("--image", default=None)
    analyze_p.add_argument("--target", type=int, default=None)
    analyze_p.add_argument("--layers", nargs="*", default=None)
    analyze_p.add_argument("--out", default="outputs")

    scan_p = sub.add_parser("dataset-scan", help="Aggregate activation sparsity over CIFAR-10")
    scan_p.add_argument("--model", default="resnet18")
    scan_p.add_argument("--layers", nargs="*", default=None)
    scan_p.add_argument("--limit", type=int, default=8)
    scan_p.add_argument("--out", default="outputs")

    list_p = sub.add_parser("list-layers", help="List layer names in order")
    list_p.add_argument("--model", default="resnet18")

    args = parser.parse_args()

    if args.cmd == "analyze":
        summary = analyze_single_image(
            model_name=args.model,
            image_path=args.image,
            target_class=args.target,
            out_dir=args.out,
            layers=args.layers,
        )
        print(json.dumps(summary, indent=2))
    elif args.cmd == "dataset-scan":
        aggregated = _dataset_scan(args.model, args.layers, args.limit, args.out)
        print(json.dumps(aggregated, indent=2))
    elif args.cmd == "list-layers":
        model = load_model(args.model, pretrained=True)
        for name, module in enumerate_layers(model):
            print(name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
