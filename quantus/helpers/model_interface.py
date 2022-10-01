from __future__ import annotations


from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class ModelInterface(ABC):
    """Interface for torch and tensorflow models."""

    def __init__(
        self,
        model,
        channel_first: bool = True,
        softmax: bool = False,
        predict_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.channel_first = channel_first
        self.softmax = softmax

        if predict_kwargs is None:
            self.predict_kwargs = {}
        else:
            self.predict_kwargs = predict_kwargs

    @abstractmethod
    def predict(self, x, **kwargs):
        """Predict on the given input."""
        raise NotImplementedError

    @abstractmethod
    def shape_input(
        self, x: np.array, shape: Tuple[int, ...], channel_first: Optional[bool] = None
    ):
        """
        Reshape input into model expected input.
        channel_first: Explicitely state if x is formatted channel first (optional).
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        """Get the original torch/tf model."""
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        """Get a dictionary of the model's learnable parameters."""
        raise NotImplementedError

    @abstractmethod
    def get_random_layer_generator(self):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        Set order to top_down for cascading randomization.
        Set order to independent for independent randomization.
        """
        raise NotImplementedError

    @abstractmethod
    def get_hidden_layers_representations(
        self,
        x: np.ndarray,
        layer_names: Optional[Tuple[str]] = None,
        layer_indices: Optional[Tuple[int]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Computes models internal representation of input x.
        In practice, this means, execute forward pass, and capture output of layers, one is interested in.
        As authors of https://arxiv.org/pdf/2203.06877.pdf did not provide neither code example
        nor details what exactly "internal model representation, e.g., output embeddings of hidden layers"
        should be, we leave it up to user whether all layers are used,
        or couple specific ones should be selected.
        User can select layer by providing 'layer_names' (exclusive)OR 'layer_indices'.

        Parameters:
            x: np.ndarray, 4D tensor, a batch of input datapoints
            layer_names: tuple with names of layers, from which output should be captured.
            layer_indices: tuple with Indices of layers, from which output should be captured. Intended to use in case, when layer names are not unique, or unknown.
        Returns:
            L(*): np.ndarray, 2D tensor with shape (batch_size, None)
        """
        pass
