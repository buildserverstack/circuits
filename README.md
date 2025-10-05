# Neural Network Activation Visualization Toolkit

This repository contains a Jupyter-first toolkit for inspecting small text and vision models. The tooling focuses on three guiding concepts:

* **Inputs → Weights → Bias → Activation → Output**: capture tensors entering each layer, measure parameter magnitudes, and surface the non-linearity effects across the network.
* **Layers (input / hidden / output)**: track activations for every major block, highlight sparsity, and summarize mean activations through a timeline strip.
* **Connections (forward / backward)**: render attention patterns, compute attributions, and export lightweight circuit graphs linking influential units.

The source code lives in `nnviz/nnviz/` and is organized by responsibility (configuration, data loading, hooks, metrics, visualization, attribution, summarization, UI, and I/O utilities). Notebooks in `nnviz/notebooks/` provide guided workflows:

1. **01_quickstart** – run the full pipeline on sample text and image inputs.
2. **02_text_model_walkthrough** – deep dive on Transformer-based language models.
3. **03_vision_model_walkthrough** – Grad-CAM and saliency overlays for vision encoders.

Assets and saved run artifacts are stored under `nnviz/assets/` and `nnviz/outputs/runs/` respectively. Refer to the notebooks for environment setup instructions and end-to-end demonstrations.
