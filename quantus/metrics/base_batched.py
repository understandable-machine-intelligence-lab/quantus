"""This module implements the base class for creating evaluation metrics."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.
from __future__ import annotations

from abc import abstractmethod, ABC
from functools import singledispatchmethod, partial
from operator import itemgetter
from typing import Callable, Dict, Optional, Any

import numpy as np
from tqdm.auto import tqdm

from quantus.functions.perturb_func import perturb_batch as perturb_batch_fn
from quantus.helpers import asserts
from quantus.helpers import warn
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.model.text_classifier import TextClassifier
from quantus.helpers.types import NormaliseFn, ExplainFn, AggregateFn
from quantus.helpers.utils import (
    infer_channel_first,
    make_channel_first,
    get_wrapped_model,
    value_or_default,
    map_optional,
    batch_inputs,
    add_default_items,
    safe_as_array
)
from quantus.helpers.nlp_utils import (
    map_explanations,
    is_plain_text_perturbation,
    is_torch_model
)
from quantus.metrics.base import EvaluateAble


class BatchedMetric(EvaluateAble, ABC):
    """
    Implementation base BatchedMetric class.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool,
            normalise: bool,
            normalise_func: Optional[NormaliseFn],
            normalise_func_kwargs: Optional[Dict[str, Any]],
            return_aggregate: bool,
            aggregate_func: Optional[AggregateFn],
            default_plot_func: Optional[Callable],
            disable_warnings: bool,
            display_progressbar: bool,
            **kwargs,
    ):
        """
        Initialise the BatchedMetric base class.

        Each of the defined metrics in Quantus, inherits from Metric or BatchedMetric base class.

        A child metric can benefit from the following class methods:
        - __call__(): Will call general_preprocess(), apply evaluate_instance() on each
                      instance and finally call custom_preprocess().
                      To use this method the child BatchedMetric needs to implement
                      evaluate_instance().
        - general_preprocess(): Prepares all necessary data structures for evaluation.
                                Will call custom_preprocess() at the end.

        Parameters
        ----------
        abs: boolean
            Indicates whether absolute operation is applied on the attribution.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call.
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed.
        kwargs: optional
            Keyword arguments.
        """

        # Initialise super-class with passed parameters.
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

    def __call__(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional,
            channel_first: Optional[bool],
            explain_func: Optional[ExplainFn],
            explain_func_kwargs: Optional[Dict[str, Any]],
            model_predict_kwargs: Optional[Dict],
            softmax: Optional[bool],
            device: Optional[str] = None,
            batch_size: int = 64,
            custom_batch: Optional[Any] = None,
            s_batch=None,
            **kwargs,
    ) -> np.ndarray:
        """
        This implementation represents the main logic of the metric and makes the class object callable.
        It completes batch-wise evaluation of explanations (a_batch) with respect to input data (x_batch),
        output labels (y_batch) and a torch or tensorflow model (model).

        Calls general_preprocess() with all relevant arguments, calls
        evaluate_instance() on each instance, and saves results to last_results.
        Calls custom_postprocess() afterwards. Finally returns last_results.

        The content of last_results will be appended to all_results (list) at the end of
        the evaluation call.

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
        custom_batch: any
            Any object that can be passed to the evaluation process.
            Gives flexibility to the user to adapt for implementing their own metric.
        kwargs: optional
            Keyword arguments.

        Returns
        -------
        last_results: list
            a list of Any with the evaluation scores of the concerned batch.

        Examples:
        --------
            # Minimal imports.
            >> import quantus
            >> from quantus import LeNet
            >> import torch

            # Enable GPU.
            >> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Load a pre-trained LeNet classification model (architecture at quantus/helpers/models).
            >> model = LeNet()
            >> model.load_state_dict(torch.load("tutorials/assets/pytests/mnist_model"))

            # Load MNIST datasets and make loaders.
            >> test_set = torchvision.datasets.MNIST(root='./sample_data', download=True)
            >> test_loader = torch.utils.data.DataLoader(test_set, batch_size=24)

            # Load a batch of inputs and outputs to use for XAI evaluation.
            >> x_batch, y_batch = iter(test_loader).next()
            >> x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()

            # Generate Saliency attributions of the test set batch of the test set.
            >> a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, abs=True).sum(axis=1)
            >> a_batch_saliency = a_batch_saliency.cpu().numpy()

            # Initialise the metric and evaluate explanations by calling the metric instance.
            >> metric = Metric(abs=True, normalise=False)
            >> scores = metric(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch_saliency}
        """
        # Run deprecation warnings.
        warn.deprecation_warnings(kwargs)
        warn.check_kwargs(kwargs)

        data = self.general_preprocess(
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            channel_first,
            explain_func,
            explain_func_kwargs,
            model_predict_kwargs,
            softmax,
            device,
            custom_batch,
            batch_size,
        )

        # We should not use un-batched version after general preprocess.
        del x_batch
        del y_batch
        del a_batch
        del s_batch
        del model

        pbar = tqdm(
            data["x_batch"],
            disable=not self.display_progressbar,
        )

        model = data["model"]
        scores_batch = []

        for i, x in enumerate(pbar):
            # Get batch from data dict.
            y = map_optional(data["y_batch"], itemgetter(i))
            a = map_optional(data["a_batch"], itemgetter(i))
            s = map_optional(data["s_batch"], itemgetter(i))
            # custom = map_optional(data["custom_batch"], itemgetter(i))
            x, y, a, custom_batch = self.batch_preprocess(model, x, y, a)
            score = self.evaluate_batch(model, x, y, a, s, custom_batch)
            score = self.batch_postprocess(model, x, y, a, s, score)
            scores_batch.extend(score)

        # Call post-processing.
        self.custom_postprocess(**data)
        if self.return_aggregate:
            scores_batch = self.aggregate_func(scores_batch)
        # Append content of last results to all results.
        scores_batch = np.asarray(scores_batch)
        self.all_results.append(scores_batch)
        return scores_batch

    @singledispatchmethod
    def explain_batch(
            self, model: ModelInterface, x_batch: np.ndarray, y_batch: np.ndarray, **kwargs
    ):
        a_batch = self.explain_func(
            model.get_model(), x_batch, y_batch, **self.explain_func_kwargs, **kwargs
        )

        if self.normalise:
            a_batch = self.normalise_func(a_batch, **self.normalise_func_kwargs)
        if self.abs:
            a_batch = np.abs(a_batch)
        return a_batch

    @explain_batch.register
    def _(
            self, model: TextClassifier, x_batch, y_batch: np.ndarray, **kwargs
    ):
        a_batch = self.explain_func(
            model, x_batch, y_batch, **self.explain_func_kwargs, **kwargs
        )
        if self.normalise:
            a_batch = map_explanations(
                a_batch, partial(self.normalise_func, **self.normalise_func_kwargs)
            )
        if self.abs:
            a_batch = map_explanations(a_batch, np.abs)

        return a_batch

    def general_preprocess(
            self,
            model,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: Optional[np.ndarray],
            channel_first: Optional[bool],
            explain_func: Callable,
            explain_func_kwargs: Optional[Dict[str, Any]],
            model_predict_kwargs: Optional[Dict],
            softmax: bool,
            device: Optional[str],
            custom_batch: Optional[np.ndarray],
            batch_size: int = 64,
    ) -> Dict[str, Any]:
        if isinstance(x_batch, np.ndarray):
            # Reshape input batch to channel first order:
            if not isinstance(channel_first, bool):  # None is not a boolean instance.
                channel_first = infer_channel_first(x_batch)
            x_batch = make_channel_first(x_batch, channel_first)
        else:
            # For NLP we don't need it
            channel_first = None

        model_wrapper = get_wrapped_model(
            model,
            channel_first,
            softmax,
            device,
            model_predict_kwargs,
        )
        # Save as attribute, some metrics need it during processing.
        self.explain_func = explain_func
        self.explain_func_kwargs = value_or_default(explain_func_kwargs, lambda: {})
        if is_torch_model(model):
            self.explain_func_kwargs = add_default_items(self.explain_func_kwargs, {"device": device})

        # Create extra axis for inputs.
        batch_fn = partial(batch_inputs, batch_size=batch_size)
        x_batch = batch_fn(x_batch)

        # Batch additional inputs, if present, otherwise keep them as None.
        # We keep y_batch and a_batch as None to avoid calling model and explain function on whole dataset at once.
        y_batch = map_optional(y_batch, batch_fn)
        a_batch = map_optional(a_batch, batch_fn)
        s_batch = map_optional(s_batch, batch_fn)
        return {
            "model": model_wrapper,
            "x_batch": x_batch,
            "y_batch": y_batch,
            "a_batch": a_batch,
            "s_batch": s_batch,
            "custom_batch": None,
        }

    def batch_preprocess(
            self,
            model,
            x_batch,
            y_batch,
            a_batch,
    ):
        # Generate y_batch if not provided.
        y_batch = value_or_default(
            y_batch, lambda: model.predict(x_batch).argmax(axis=-1)
        )
        # Generate a_batch if not provided.
        a_batch = value_or_default(
            a_batch,
            lambda: self.explain_batch(
                model,
                x_batch,
                y_batch,
            ),
        )
        return x_batch, y_batch, a_batch, None

    def batch_postprocess(
            self,
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            score,
    ) -> np.ndarray:
        return score

    @abstractmethod
    def evaluate_batch(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: np.ndarray,
            a_batch: np.ndarray,
            s_batch: np.ndarray,
            custom_batch=None,
    ):
        """
        Evaluates model and attributes on a single data batch and returns the batched evaluation result.

        This method needs to be implemented to use __call__().

        Parameters
        ----------
        model: ModelInterface
            A ModelInteface that is subject to explanation.
        x_batch: np.ndarray
            The input to be evaluated on a batch-basis.
        y_batch: np.ndarray
            The output to be evaluated on a batch-basis.
        a_batch: np.ndarray
            The explanation to be evaluated on a batch-basis.
        s_batch: np.ndarray
            The segmentation to be evaluated on a batch-basis.
        custom_batch:


        Returns
        -------
        np.ndarray
            The batched evaluation results.
        """
        raise NotImplementedError()


class BatchedPerturbationMetric(BatchedMetric, ABC):
    """
    Implementation base BatchedPertubationMetric class.

    This batched metric has additional attributes for perturbations.
    """

    @asserts.attributes_check
    def __init__(
            self,
            abs: bool,
            normalise: bool,
            normalise_func: Optional[Callable],
            normalise_func_kwargs: Optional[Dict[str, Any]],
            perturb_func: Callable,
            perturb_func_kwargs: Optional[Dict[str, Any]],
            return_aggregate: bool,
            aggregate_func: Optional[Callable],
            default_plot_func: Optional[Callable],
            disable_warnings: bool,
            display_progressbar: bool,
            nr_samples: int = None,
            return_nan_when_prediction_changes: bool = None,
            **kwargs,
    ):
        """
        Initialise the PerturbationMetric base class.

        Parameters
        ----------
        abs: boolean
            Indicates whether absolute operation is applied on the attribution.
        normalise: boolean
            Indicates whether normalise operation is applied on the attribution.
        normalise_func: callable
            Attribution normalisation function applied in case normalise=True.
        normalise_func_kwargs: dict
            Keyword arguments to be passed to normalise_func on call.
        perturb_func: callable
            Input perturbation function.
        perturb_func_kwargs: dict
            Keyword arguments to be passed to perturb_func, default={}.
        return_aggregate: boolean
            Indicates if an aggregated score should be computed over all instances.
        aggregate_func: callable
            Callable that aggregates the scores given an evaluation call..
        default_plot_func: callable
            Callable that plots the metrics result.
        disable_warnings: boolean
            Indicates whether the warnings are printed.
        display_progressbar: boolean
            Indicates whether a tqdm-progress-bar is printed.
        kwargs: optional
            Keyword arguments.
        """

        # Initialise super-class with passed parameters.
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            default_plot_func=default_plot_func,
            display_progressbar=display_progressbar,
            disable_warnings=disable_warnings,
            **kwargs,
        )

        # Save perturbation metric attributes.
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = value_or_default(perturb_func_kwargs, lambda: {})
        self.return_nan_when_prediction_changes = return_nan_when_prediction_changes
        self.nr_samples = nr_samples

    @singledispatchmethod
    def changed_prediction_indices(self, model, x_batch, x_perturbed, **kwargs):
        if not self.return_nan_when_prediction_changes:
            return []

        # do we need it ???
        # x_input = model.shape_input(
        #    x=x_perturbed,
        #    shape=x_batch.shape,
        #    channel_first=True,
        #    batched=True,
        # )
        og_labels = model.predict(x_batch).argmax(axis=-1)
        perturbed_labels = model.predict(x_perturbed).argmax(axis=-1)
        return np.reshape(np.argwhere(og_labels != perturbed_labels), -1)

    @changed_prediction_indices.register
    def _(self, model: TextClassifier, x_batch, x_perturbed, **kwargs):
        if not self.return_nan_when_prediction_changes:
            return []

        if is_plain_text_perturbation(self.perturb_func):
            og_logits = model.predict(x_batch)
            perturbed_logits = model.predict(x_perturbed)
        else:
            og_logits = safe_as_array(model(x_batch, **kwargs))
            perturbed_logits = safe_as_array(model(x_perturbed, **kwargs))

        og_labels = np.argmax(og_logits, axis=-1)
        perturbed_labels = np.argmax(perturbed_logits, axis=-1)

        return np.reshape(np.argwhere(og_labels != perturbed_labels), -1)

    def perturb_batch(self, x_batch: np.ndarray) -> np.ndarray:

        batch_size = x_batch.shape[0]
        size = np.size(x_batch[0])
        ndim = np.ndim(x_batch[0])

        return perturb_batch_fn(
            perturb_func=self.perturb_func,
            indices=np.tile(np.arange(0, size), (batch_size, 1)),
            indexed_axes=np.arange(0, ndim),
            arr=x_batch,
            **self.perturb_func_kwargs,
        )

    def custom_preprocess(
            self,
            model: ModelInterface,
            x_batch: np.ndarray,
            y_batch: Optional[np.ndarray],
            a_batch: Optional[np.ndarray],
            s_batch: np.ndarray,
            custom_batch: Optional[np.ndarray],
    ) -> None:
        """
        Implementation of custom_preprocess_batch.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A torch or tensorflow model e.g., torchvision.models that is subject to explanation.
        x_batch: np.ndarray
            A np.ndarray which contains the input data that are explained.
        y_batch: np.ndarray
            A np.ndarray which contains the output labels that are explained.
        a_batch: np.ndarray, optional
            A np.ndarray which contains pre-computed attributions i.e., explanations.
        s_batch: np.ndarray, optional
            A np.ndarray which contains segmentation masks that matches the input.
        custom_batch: any
            Gives flexibility ot the user to use for evaluation, can hold any variable.

        Returns
        -------
        None
        """
        # Additional explain_func assert, as the one in prepare() won't be
        # executed when a_batch != None.
        asserts.assert_explain_func(explain_func=self.explain_func)

    def batch_preprocess(
            self,
            model,
            x_batch,
            y_batch,
            a_batch,
    ):
        if "NLP" not in self.data_domain_applicability:
            return super().batch_preprocess(model, x_batch, y_batch, a_batch)
        if not is_plain_text_perturbation(self.perturb_func):
            return super().batch_preprocess(model, x_batch, y_batch, a_batch)

        batch_size = len(x_batch)
        # For plain text we need to first collect perturbations, then
        # "pre-tokenize" them, so we end up with sequences all the same length.
        x_perturbed_batches = [x_batch] + [
            self.perturb_func(x_batch, **self.perturb_func_kwargs)
            for _ in range(self.nr_samples)
        ]
        x_perturbed_batches = np.reshape(x_perturbed_batches, -1).tolist()
        x_perturbed_ids, _ = model.tokenizer.get_input_ids(x_perturbed_batches)
        x_perturbed_batches = model.tokenizer.batch_decode(x_perturbed_ids)
        x_batch, x_perturbed_batches = (
            x_perturbed_batches[:batch_size],
            x_perturbed_batches[batch_size:],
        )
        x_perturbed_batches = np.reshape(x_perturbed_batches, (self.nr_samples, -1))
        x_perturbed_batches = [i.tolist() for i in x_perturbed_batches]

        # Leave padding tokens in.
        model.tokenizer.batch_encode = partial(
            model.tokenizer.batch_encode, add_special_tokens=False
        )
        x_batch, y_batch, a_batch, _ = super().batch_preprocess(
            model, x_batch, y_batch, a_batch
        )
        return x_batch, y_batch, a_batch, x_perturbed_batches

    def batch_postprocess(
            self,
            model,
            x_batch,
            y_batch,
            a_batch,
            s_batch,
            score: np.ndarray,
    ) -> np.ndarray:
        if isinstance(x_batch[0], str):
            if is_plain_text_perturbation(self.perturb_func):
                model.tokenizer.batch_encode = model.tokenizer.batch_encode.func
        return super().batch_postprocess(
            model, x_batch, y_batch, a_batch, s_batch, score
        )
