"""This module contains the implementation of the Pixel-Flipping metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from quantus.helpers import asserts
from quantus.helpers import plotting
from quantus.helpers import utils
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.functions.normalise_func import normalise_by_max
from quantus.functions.perturb_func import baseline_replacement_by_indices
from quantus.metrics.base import Metric
from quantus.helpers.enums import (
    ModelType,
    DataType,
    ScoreDirection,
    EvaluationCategory,
)


class InverseEstimation(Metric):
    """
    Implementation of Inverse Estimation experiment by Author et al., 2023.

    The basic idea is to ..............

    References:
        1) ..........

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
        - evaluation_category: What property/ explanation quality that this metric measures.
    """

    name = "Inverse-Estimation"
    data_applicability = {DataType.IMAGE, DataType.TIMESERIES, DataType.TABULAR}
    model_applicability = {ModelType.TORCH, ModelType.TF}
    score_direction = ScoreDirection.HIGHER
    # evaluation_category = EvaluationCategory.FAITHFULNESS

    def __init__(
        self,
        metric_init: Metric,
        inverse_method: str = "sign-flip",
        return_mean_per_sample: bool = True,
        return_auc_per_sample: bool = False,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[Callable] = None,
        normalise_func_kwargs: Optional[Dict[str, Any]] = None,
        return_aggregate: Optional[bool] = True,
        aggregate_func: Optional[Callable] = None,
        default_plot_func: Optional[Callable] = None,
        disable_warnings: Optional[bool] = None,
        display_progressbar: Optional[bool] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        metric_init: Metric
            The metric to be used for the inverse estimation.
        abs: boolean
            Indicates whether absolute operation is applied on the attribution, default=False.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution, default=True.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
            If normalise_func=None, the default value is used, default=normalise_by_max.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call, default={}.
        perturb_func: callable
            Input perturbation function. If None, the default value is used,
            default=baseline_replacement_by_indices.
        perturb_baseline: string
            Indicates the type of baseline: "mean", "random", "uniform", "black" or "white",
            default="black".
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        return_auc_per_sample: boolean
            Indicates if an AUC score should be computed over the curve and returned.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed, default=False.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed, default=False.
        kwargs: optional
            Keyword arguments.
        """
        if metric_init.default_plot_func is None:
            default_plot_func = plotting.plot_inverse_curves

        self.return_aggregate = return_aggregate

        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=self.return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        self.return_auc_per_sample = return_auc_per_sample
        self.return_mean_per_sample = return_mean_per_sample
        self.inverse_method = inverse_method
        self.metric_init = metric_init

        # TODO. Update warnings.
        assert not (
            self.return_mean_per_sample and self.return_auc_per_sample
        ), "Only one of 'return_mean_per_sample' and 'return_auc_per_sample' can be True."
        if self.inverse_method not in ["sign-flip", "value-swap"]:
            raise ValueError(
                "The 'inverse_method' in init **kwargs, \
                             must be either 'sign-flip' or 'value-swap'."
            )
        if self.metric_init.return_aggregate:
            print(
                "The metric is not designed to return an aggregate score, setting return_aggregate=False."
            )
            self.metric_init.return_aggregate = False

        assert self.metric_init.abs == False, (
            "To run the inverse estimation, you cannot set 'a_batch' to "
            "have positive attributions only. Set 'abs' param of the metric init to 'False'."
        )

        if not self.disable_warnings:
            warn.warn_parameterisation(
                metric_name=self.__class__.__name__,
                sensitive_params=("baseline value 'perturb_baseline'"),
                citation=("Update here."),
            )

    def __call__(
        self,
        model,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Optional[np.ndarray] = None,
        s_batch: Optional[np.ndarray] = None,
        channel_first: Optional[bool] = True,
        explain_func: Optional[Callable] = None,
        explain_func_kwargs: Optional[Dict] = None,
        model_predict_kwargs: Optional[Dict] = None,
        softmax: Optional[bool] = True,
        device: Optional[str] = None,
        batch_size: int = 64,
        custom_batch: Optional[Any] = None,
        **kwargs,
    ) -> List[float]:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes instance-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        () on each instance, and saves results to evaluation_scores.
        Calls custom_postprocess() afterwards. Finally returns evaluation_scores.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        explain_func: callable
            Callable generating attributions.
        explain_func_kwargs: dict, optional
            Keyword arguments to be passed to explain_func on call.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        evaluation_scores: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >>> import quantus
            >>> from quantus import LeNet
            >>> import torch

            # Enable GPU.
            >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >>> model = LeNet()
            >>> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >>> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >>> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >>> x_batch, y_batch = iter(test_loader).next()
            >>> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >>> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >>> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >>> metric = Metric(abs=True, normalise=False)
            >>> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """
        return super().__call__(
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            custom_batch=None,
            channel_first=channel_first,
            explain_func=explain_func,
            explain_func_kwargs=explain_func_kwargs,
            model_predict_kwargs=model_predict_kwargs,
            softmax=softmax,
            device=device,
            batch_size=batch_size,
            **kwargs,
        )

    def evaluate_batch(
        self,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        s_batch: Optional[np.ndarray] = None,
        **kwargs,
    ) -> List[float]:

        assert (
            a_batch is not None
        ), "'a_batch' must be provided to run the inverse estimation."

        # Metrics that depend on re-computing explanations need inverse wrapping.
        self.explain_func_kwargs["explain_func"] = self.explain_func
        self.explain_func_kwargs["inverse_method"] = self.inverse_method

        self.scores_ori = self.metric_init(
            model=model.get_model(),
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            channel_first=self.channel_first,
            explain_func=self.explain_func,
            explain_func_kwargs=self.explain_func_kwargs,
            softmax=self.softmax,
            device=self.device,
            model_predict_kwargs=self.model_predict_kwargs,
            **kwargs,
        )
        assert len(self.scores_ori) == len(x_batch), (
            "To run the inverse estimation, the number of evaluation scores "
            "must match the number of instances in the batch."
        )

        # Empty the evaluation scores before re-scoring with the metric.
        self.metric_init.evaluation_scores = []

        # Get inverse attributions.
        a_batch_inv = self.get_inverse_attributions(a_batch=a_batch)

        # Run inverse experiment.
        self.scores_inv = self.metric_init(
            model=model.get_model(),
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch_inv,
            s_batch=s_batch,
            channel_first=self.channel_first,
            explain_func=self.inverse_explain_wrapper,
            explain_func_kwargs=self.explain_func_kwargs,
            softmax=self.softmax,
            device=self.device,
            model_predict_kwargs=self.model_predict_kwargs,
            **kwargs,
        )

        # Compute the inverse, empty the evaluation scores again and overwrite with the inverse scores.
        inv_scores = np.array(self.scores_ori) - np.array(self.scores_inv)
        print("Scores shape", np.shape(self.scores_ori), np.shape(self.scores_inv))
        print("Inverse shape", np.shape(inv_scores))
        print("Inverse shape", np.reshape(inv_scores, (len(inv_scores), -1)).shape)
        if self.return_mean_per_sample:
            inv_scores = self.get_mean_score(scores=inv_scores)
            print("Agg shape", np.shape(inv_scores))
        elif self.return_auc_per_sample:
            inv_scores = self.get_auc_score(scores=inv_scores)

        return inv_scores.tolist()

    def get_mean_score(self, scores):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return np.mean(np.array(scores), axis=1)

    def get_auc_score(self, scores):
        """Calculate the area under the curve (AUC) score for several test samples."""
        return np.mean([utils.calculate_auc(np.array(curve)) for curve in scores])

    def get_inverse_attributions(self, a_batch: np.array):
        """Get the inverse attributions of the input attributions."""

        # Attributions need to have only one axis, else flatten and reshape back.
        shape_ori = a_batch.shape
        a_batch = a_batch.reshape((shape_ori[0], -1))
        if self.inverse_method == "sign-flip":
            a_batch_inv = -np.array(a_batch)
        elif self.inverse_method == "value-swap":
            indices = np.argsort(a_batch, axis=1)
            a_batch_inv = np.empty_like(a_batch)
            a_batch_inv[np.arange(a_batch_inv.shape[0])[:, None], indices] = a_batch[
                np.arange(a_batch_inv.shape[0])[:, None], indices[:, ::-1]
            ]
        a_batch_inv = a_batch_inv.reshape(shape_ori)
        return a_batch_inv

    def inverse_explain_wrapper(self, model, inputs, targets, **kwargs):
        """Wrapper for the explanation function that computes the inverse attributions."""
        a_batch = self.explain_func(model, inputs, targets, **self.explain_func_kwargs)
        a_batch_inv = self.get_inverse_attributions(a_batch=a_batch)
        return a_batch_inv
