from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Generator

import numpy as np
import torch
import torch.nn as nn

from quantus.nlp.helpers.model.text_classifier import TextClassifier


class TorchTextClassifier(TextClassifier):
    def get_random_layer_generator(
        self, order: str = "top_down", seed: int = 42
    ) -> Generator:
        original_parameters = self.weights
        model_copy = self.clone()

        modules = [
            layer
            for layer in self.unwrap().named_modules()
            if (hasattr(layer[1], "reset_parameters") and len(layer.state_dict()) > 0)
        ]

        if order == "top_down":
            modules = modules[::-1]

        for module in modules:
            if order == "independent":
                model_copy.weights = original_parameters
            torch.manual_seed(seed=seed + 1)
            module[1].reset_parameters()

            yield module[0], model_copy

    @property
    def random_layer_generator_length(self) -> int:
        modules = [
            layer
            for layer in self.unwrap().named_modules()
            if (hasattr(layer[1], "reset_parameters") and len(layer.state_dict()) > 0)
        ]
        return len(modules)

    def to_tensor(self, x: torch.Tensor | np.ndarray, **kwargs) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, device=self.device, **kwargs)

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        return self.unwrap().state_dict()

    @weights.setter
    def weights(self, weights: Dict[str, torch.Tensor]):
        self.unwrap().load_state_dict(weights)

    @abstractmethod
    def unwrap(self) -> nn.Module:
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> TorchTextClassifier:
        raise NotImplementedError
