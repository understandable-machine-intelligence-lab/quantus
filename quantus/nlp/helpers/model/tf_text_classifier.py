# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from abc import abstractmethod
from typing import Generator, List

import numpy as np
import tensorflow as tf

from quantus.nlp.helpers.model.text_classifier import TextClassifier


class TensorFlowTextClassifier(TextClassifier, tf.Module):
    def get_random_layer_generator(
        self,
        order: str = "top_down",
        seed: int = 42,
    ) -> Generator:
        original_weights = self.weights.copy()
        model_copy = self.clone()
        layers = list(
            model_copy.unwrap()._flatten_layers(  # noqa
                include_self=False, recursive=True
            )
        )
        layers = list(filter(lambda i: len(i.get_weights()) > 0, layers))

        if order == "top_down":
            layers = layers[::-1]

        for layer in layers:
            if order == "independent":
                model_copy.weights = original_weights
            weights = layer.get_weights()
            np.random.seed(seed=seed + 1)
            layer.set_weights([np.random.permutation(w) for w in weights])
            yield layer.name, model_copy

    @property
    def random_layer_generator_length(self) -> int:
        layers = list(
            self.unwrap()._flatten_layers(include_self=False, recursive=True)  # noqa
        )
        layers = list(filter(lambda i: len(i.get_weights()) > 0, layers))
        return len(layers)

    @property
    def weights(self) -> List[np.ndarray]:
        return self.unwrap().get_weights()

    @weights.setter
    def weights(self, weights: List[np.ndarray]):
        self.unwrap().set_weights(weights)

    @property
    @abstractmethod
    def unwrap(self) -> tf.keras.Model:
        raise NotImplementedError

    @abstractmethod
    def clone(self) -> TensorFlowTextClassifier:
        raise NotImplementedError
