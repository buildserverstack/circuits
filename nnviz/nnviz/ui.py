"""Build ipywidgets controls for model/layer/head selection, thresholds, and buttons. Wire callbacks to re-render plots without re-running the full forward unless inputs changed."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import ipywidgets as widgets

from .config import SUPPORTED_TEXT_MODELS, SUPPORTED_VISION_MODELS


class ControlPanel:
    """Assemble UI controls for the notebook experience."""

    def __init__(
        self,
        on_forward: Callable[[], None],
        on_backward: Callable[[], None],
        on_refresh: Callable[[], None],
    ) -> None:
        self.model_dropdown = widgets.Dropdown(
            options=SUPPORTED_TEXT_MODELS + SUPPORTED_VISION_MODELS,
            description="Model",
        )
        self.layer_dropdown = widgets.Dropdown(options=[], description="Layer")
        self.head_dropdown = widgets.IntSlider(min=0, max=0, step=1, description="Head")
        self.threshold_slider = widgets.FloatSlider(
            value=0.05, min=0.0, max=0.3, step=0.01, description="Threshold"
        )
        self.topk_slider = widgets.IntSlider(value=20, min=1, max=64, description="Top-k Units")
        self.forward_button = widgets.Button(description="Run Forward", button_style="success")
        self.backward_button = widgets.Button(description="Run Backward", button_style="warning")
        self.save_button = widgets.Button(description="Save Run", button_style="")

        self.forward_button.on_click(lambda _: on_forward())
        self.backward_button.on_click(lambda _: on_backward())
        self.save_button.on_click(lambda _: on_refresh())

    def as_widget(self) -> widgets.VBox:
        return widgets.VBox(
            [
                self.model_dropdown,
                self.layer_dropdown,
                self.head_dropdown,
                self.threshold_slider,
                self.topk_slider,
                widgets.HBox([self.forward_button, self.backward_button, self.save_button]),
            ]
        )


__all__ = ["ControlPanel"]
