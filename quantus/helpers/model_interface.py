from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np


class ModelInterface(ABC):
    """Interface for torch and tensorflow models."""

    def __init__(self, model, channel_first=True):
        self.model = model
        self.channel_first = channel_first

    @abstractmethod
    def predict(self, x_input):
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
    def get_hidden_layers_outputs(self,
                                  x: np.ndarray,
                                  layer_names: Optional[List[str]] = None,
                                  layer_indices: Optional[List[int]] = None,
                                  ) -> np.ndarray:
        """
        If no layer_names and no layer_indices provided, collect all hidden layers outputs
        (aka intermediate representations), and concatenate them in 1D tensor.
        Alternatively, user can provide names or indices of layers of interest.
        """
        pass
