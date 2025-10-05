# NNViz: Neuron & Activation Visualization for PyTorch

## What It Does

* Captures **pre/post activations**, **gradients**, and **weights**.
* Visualizes **activated vs unactivated neurons**, **feature maps**, **saliency**, **Grad-CAM**, and a **network graph** with weighted edges.

## Install (Conda + JupyterLab)

```bash
# 1) Create env
conda env create -f environment.yml
conda activate nnviz

# 2) (Optional) Verify CPU-only torch is selected
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 3) Launch JupyterLab
jupyter lab
```

## Quick Start (CLI)

```bash
# Analyze a single image with ResNet18
python -m src.cli analyze \
  --model resnet18 \
  --image samples/dog.jpg \
  --target 243 \
  --layers layer1.0.relu layer2.0.relu \
  --out outputs

# List available layers for targeting Grad-CAM
python -m src.cli list-layers --model resnet18
```

Artifacts will appear in `outputs/`:

* `network_graph.png` (layers + weighted edges)
* `feature_maps_*.png`
* `activation_hist_*.png` (activated vs unactivated, with sparsity)
* `neuron_grid_fc.png` (weights/bias per neuron)
* `saliency.png`, `grad_cam_*.png`

## Quick Start (Notebook)

Open `notebooks/00_quickstart.ipynb` and run all cells.

## How We Represent Concepts

* **Inputs (features):** normalized tensors.
* **Weights (importance):** heatmaps; edge thickness in graph.
* **Bias:** per-neuron scalar labels.
* **Activation function:** pre/post plots; sparsity report.
* **Forward pass:** compute/store activations.
* **Backward pass:** Captum saliency; Grad-CAM.

## Add Your Model

```python
from src.models import load_model
model = load_model("mobilenet_v3_small", pretrained=True)
```

## Troubleshooting

* Out-of-memory: use fewer layers; disable storing `pre` activations; set `max_batches=1`.
* Matplotlib blank images: ensure `plt.close()` after each save.
