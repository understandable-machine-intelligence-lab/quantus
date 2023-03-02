"""Explanation functions for TensorFlow models."""
from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Callable, Optional, Dict, Union
from functools import singledispatch

from quantus.nlp.helpers.model.tensorflow_text_classifier import (
    TensorFlowTextClassifier,
)
from quantus.nlp.helpers.types import (
    Explanation,
)
from quantus.nlp.helpers.utils import (
    value_or_default,
    get_embeddings,
    get_input_ids,
    apply_noise,
    get_interpolated_inputs,
    get_logits_for_labels,
    tf_function,
    map_dict,
)
import numpy as np


# Just to save some typing effort
_BaselineFn = Callable[[tf.Tensor], tf.Tensor]
_TextOrVector = Union[List[str], tf.Tensor]
_Scores = Union[List[Explanation], tf.Tensor]
_NoiseFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

# ----------------- "Entry Point" --------------------


def available_explanation_functions() -> Dict[str, Callable]:
    return {
        "GradNorm": explain_gradient_norm,
        "GradXInput": explain_gradient_x_input,
        "IntGrad": explain_integrated_gradients,
        "NoiseGrad": explain_noise_grad,
        "NoiseGrad++": explain_noise_grad_plus_plus,
    }


def tf_explain(
    *args,
    method: str,
    **kwargs,
) -> _Scores:
    """Execute gradient based explanation method."""

    method_mapping = available_explanation_functions()

    if method not in available_explanation_functions():
        raise ValueError(
            f"Unsupported explanation method: {method}, supported are: {list(method_mapping.keys())}"
        )
    explain_fn = method_mapping[method]
    return explain_fn(*args, **kwargs)  # noqa


# --------- "API" -----------


def explain_gradient_norm(
    model: TensorFlowTextClassifier,
    x_batch: _TextOrVector,
    y_batch: np.ndarray,
    **kwargs,
) -> _Scores:
    """
    A baseline GradientNorm text-classification explainer.
    The implementation is based on https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L38.
    GradientNorm explanation algorithm is:
        - Convert inputs to models latent representations.
        - Execute forwards pass
        - Retrieve logits for y_batch.
        - Compute gradient of logits with respect to input embeddings.
        - Compute L2 norm of gradients.

    References:
    ----------
    - https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L38

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    return sd_explain_gradient_norm(x_batch, model, y_batch, **kwargs)


def explain_gradient_x_input(
    model: TensorFlowTextClassifier,
    x_batch: _TextOrVector,
    y_batch: np.ndarray,
    **kwargs,
) -> _Scores:
    """
    A baseline GradientXInput text-classification explainer.
     The implementation is based on https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L108.
     GradientXInput explanation algorithm is:
        - Convert inputs to models latent representations.
        - Execute forwards pass
        - Retrieve logits for y_batch.
        - Compute gradient of logits with respect to input embeddings.
        - Compute vector dot product between input embeddings and gradients.


    References:
    ----------
    - https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L108

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    return sd_explain_gradient_x_input(x_batch, model, y_batch, **kwargs)


def explain_integrated_gradients(
    model: TensorFlowTextClassifier,
    x_batch: _TextOrVector,
    y_batch: np.ndarray,
    num_steps: int = 10,
    baseline_fn: Optional[_BaselineFn] = None,
    batch_interpolated_inputs: bool = False,
    **kwargs,
) -> _Scores:
    """
    A baseline Integrated Gradients text-classification explainer. Integrated Gradients explanation algorithm is:
        - Convert inputs to models latent representations.
        - For each x, y in x_batch, y_batch
        - Generate num_steps samples interpolated from baseline to x.
        - Execute forwards pass.
        - Retrieve logits for y.
        - Compute gradient of logits with respect to interpolated samples.
        - Estimate integral over interpolated samples using trapezoid rule.
    In practise, we combine all interpolated samples in one batch, to avoid executing forward and backward passes
    in for-loop. This means potentially, that batch size selected for this XAI method should be smaller than usual.

    References:
    ----------
    - https://github.com/PAIR-code/lit/blob/main/lit_nlp/components/gradient_maps.py#L108
    - Sundararajan et al., 2017, Axiomatic Attribution for Deep Networks, https://arxiv.org/pdf/1703.01365.pdf

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    num_steps:
        Number of interpolated samples, which should be generated, default=10.
    baseline_fn:
        Function used to created baseline values, by default will create zeros tensor. Alternatively, e.g.,
        embedding for [UNK] token could be used.
    batch_interpolated_inputs:
        Indicates if interpolated inputs should be stacked into 1 bigger batch.
        This speeds up the explanation, however can be very memory intensive.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    Examples
    -------
    Specifying [UNK] token as baseline:

    >>> def unknown_token_baseline_function(x):
        ... return tf.convert_to_tensor(np.load(...), dtype=tf.float32)

    >>> tf_explain_integrated_gradients(..., ..., ..., baseline_fn=unknown_token_baseline_function) # noqa

    """
    return sd_explain_integrated_gradients(
        x_batch,
        model,
        y_batch,
        num_steps=num_steps,
        baseline_fn=baseline_fn,
        batch_interpolated_inputs=batch_interpolated_inputs,
    )


def explain_noise_grad(
    model: TensorFlowTextClassifier,
    x_batch: _TextOrVector,
    y_batch: np.ndarray,
    *,
    n: int = 10,
    mean: float = 1.0,
    std: float = 0.2,
    explain_fn: Union[Callable, str] = "IntGrad",
    noise_type: str = "multiplicative",
    seed: int = 42,
    **kwargs,
) -> _Scores:
    """
    NoiseGrad++ is a state-of-the-art gradient based XAI method, which enhances baseline explanation function
    by adding stochasticity to model's weights. The implementation is based
    on https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    mean:
        Mean of normal distribution, from which noise applied to model's weights is sampled, default=1.0.
    std:
        Standard deviation of normal distribution, from which noise applied to model's weights is sampled, default=0.2.
    n:
        Number of times noise is applied to weights, default=10.
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad.
        Otherwise, must have `Callable[[TensorFlowTensorFlowTextClassifier, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]` signature.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
        Default IntGrad.
    noise_type:
        If NoiseType.multiplicative weights and input embeddings will be multiplied by noise.
        If NoiseType.additive noise will be added to weights and input embeddings.

    seed:
        PRNG seed used for noise generating distributions.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.


    Examples
    -------
    Passing kwargs to baseline explanation function:

    >>> import functools
    >>> explain_fn = functools.partial(tf_explain_integrated_gradients_numerical, num_steps=22) # noqa
    >>> tf_explain_noise_grad_plus_plus(..., ..., ..., explain_fn=explain_fn) # noqa

    References
    -------
    - https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.
    - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185

    """
    return sd_explain_noise_grad(
        x_batch,
        model,
        y_batch,
        n=n,
        mean=mean,
        std=std,
        explain_fn=explain_fn,
        noise_type=noise_type,
        seed=seed,
        **kwargs,
    )


def explain_noise_grad_plus_plus(
    model: TensorFlowTextClassifier,
    x_batch: _TextOrVector,
    y_batch: p.ndarray,
    *,
    n: int = 10,
    m: int = 10,
    mean: float = 1.0,
    std: float = 0.2,
    sg_mean: float = 0.0,
    sg_std: float = 0.4,
    explain_fn: Union[Callable, str] = "IntGrad",
    noise_type: str = "multiplicative",
    seed: int = 42,
    **kwargs,
) -> _Scores:
    """
    NoiseGrad++ is a state-of-the-art gradient based XAI method, which enhances baseline explanation function
    by adding stochasticity to model's weights and model's inputs. The implementation is based
    on https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.

    Parameters
    ----------
    model:
        A model, which is subject to explanation.
    x_batch:
        A batch of plain text inputs, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    mean:
        Mean of normal distribution, from which noise applied to model's weights is sampled, default=1.0.
    std:
        Standard deviation of normal distribution, from which noise applied to model's weights is sampled, default=0.2.
    sg_mean:
        Mean of normal distribution, from which noise applied to input embeddings is sampled, default=0.0.
    sg_std:
        Standard deviation of normal distribution, from which noise applied to input embeddings is sampled, default=0.4.
    n:
        Number of times noise is applied to weights, default=10.
    m:
        Number of times noise is applied to input embeddings, default=10
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad.
        Otherwise, must have `Callable[[TensorFlowTensorFlowTextClassifier, np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]` signature.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
        Default IntGrad.
    noise_type:
        If NoiseType.multiplicative weights and input embeddings will be multiplied by noise.
        If NoiseType.additive noise will be added to weights and input embeddings.

    seed:
        PRNG seed used for noise generating distributions.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.


    Examples
    -------
    Passing kwargs to baseline explanation function:

    >>> import functools
    >>> explain_fn = functools.partial(tf_explain_integrated_gradients_numerical, num_steps=22) # noqa
    >>> tf_explain_noise_grad_plus_plus(..., ..., ..., explain_fn=explain_fn) # noqa

    References
    -------
    - https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.
    - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185

    """
    return sd_explain_noise_grad_plus_plus(
        x_batch,
        model,
        y_batch,
        n=n,
        m=m,
        mean=mean,
        sg_mean=sg_mean,
        std=std,
        sg_std=sg_std,
        explain_fn=explain_fn,
        noise_type=noise_type,
        seed=seed,
        **kwargs,
    )


# --------- Single dispatch generic function stubs -----------
# We need those because functools.singledispatch supports only dipatching based on 1st argument type,
# which in our case is model, so we need to reorder them, so x_batch (text or embedding) is in 1st place,
# and we and up dispatching to different functions based on input type. (sd_ in name stand for single dispatch.)


@singledispatch
def sd_explain_gradient_norm(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def sd_explain_gradient_x_input(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def sd_explain_integrated_gradients(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def sd_explain_noise_grad(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def sd_explain_noise_grad_plus_plus(
    x_batch,
    *args,
    **kwargs,
) -> _Scores:
    pass


# ----------------------- GradNorm -------------------------


@sd_explain_gradient_norm.register
def _(
    x_batch: list, model: TensorFlowTextClassifier, y_batch: np.ndarray, **kwargs
) -> List[Explanation]:
    input_ids, _ = get_input_ids(x_batch, model)
    embeddings, kwargs = get_embeddings(x_batch, model)
    scores = sd_explain_gradient_norm(embeddings, model, y_batch, **kwargs)
    # TODO Vectorized map ???
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@sd_explain_gradient_norm.register(tf.Tensor)
@tf_function
def _(
    x_batch: tf.Tensor,
    model: TensorFlowTextClassifier,
    y_batch: np.ndarray,
    **kwargs,
) -> np.ndarray:
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(x_batch, **kwargs)
        logits_for_label = get_logits_for_labels(logits, y_batch)

    grads = tape.gradient(logits_for_label, x_batch)
    return tf.linalg.norm(grads, axis=-1)


# ----------------------- GradXInput -------------------------


@sd_explain_gradient_x_input.register
def _(
    x_batch: list, model: TensorFlowTextClassifier, y_batch: np.ndarray, **kwargs
) -> List[Explanation]:
    input_ids, _ = get_input_ids(x_batch, model)
    embeddings, kwargs = get_embeddings(x_batch, model)
    scores = sd_explain_gradient_x_input(embeddings, model, y_batch, **kwargs)

    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@sd_explain_gradient_x_input.register(tf.Tensor)
@tf_function
def _(
    x_batch: tf.Tensor,
    model: TensorFlowTextClassifier,
    y_batch: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """A version of GradientXInput explainer meant for usage together with latent space perturbations and/or NoiseGrad++ explainer."""

    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(x_batch, **kwargs)
        logits_for_label = get_logits_for_labels(logits, y_batch)

    grads = tape.gradient(logits_for_label, x_batch)
    return tf.math.reduce_sum(x_batch * grads, axis=-1)


# ----------------------- IntGrad -------------------------


@sd_explain_integrated_gradients.register
def _(
    x_batch: list,
    model: TensorFlowTextClassifier,
    y_batch: np.ndarray,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[_BaselineFn] = None,
    batch_interpolated_inputs: bool = False,
    **kwargs,
) -> List[Explanation]:
    input_ids, _ = get_input_ids(x_batch, model)
    embeddings, kwargs = get_embeddings(x_batch, model)
    scores = sd_explain_integrated_gradients(
        embeddings,
        model,
        y_batch,
        num_steps=num_steps,
        baseline_fn=baseline_fn,
        batch_interpolated_inputs=batch_interpolated_inputs,
        **kwargs,
    )

    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@sd_explain_integrated_gradients.register
def _(
    x_batch: tf.Tensor,
    model: TensorFlowTextClassifier,
    y_batch: np.ndarray,
    *,
    num_steps: int = 10,
    baseline_fn: Optional[_BaselineFn] = None,
    batch_interpolated_inputs: bool = False,
    **kwargs,
) -> np.ndarray:
    """A version of Integrated Gradients explainer meant for usage together with latent space perturbations and/or NoiseGrad++ explainer."""

    baseline_fn = value_or_default(
        baseline_fn, lambda: lambda x: tf.zeros_like(x, dtype=x_batch.dtype)
    )
    interpolated_embeddings = tf.map_fn(
        lambda i: get_interpolated_inputs(baseline_fn(i), i, num_steps), x_batch
    )
    if batch_interpolated_inputs:
        return _integrated_gradients_batched(
            model, interpolated_embeddings, y_batch, **kwargs
        )
    else:
        return _integrated_gradients_iterative(
            model, interpolated_embeddings, y_batch, **kwargs
        )


@tf_function
def _integrated_gradients_batched(
    model: TensorFlowTextClassifier,
    interpolated_embeddings: List[_TF_TensorLike],
    y_batch: _TF_TensorLike,
    **kwargs,
) -> np.ndarray:
    interpolated_embeddings = tf.convert_to_tensor(interpolated_embeddings)
    num_steps = interpolated_embeddings.shape[1]  # type: ignore

    interpolated_embeddings = tf.reshape(
        tf.cast(interpolated_embeddings, dtype=tf.float32),
        [-1, *interpolated_embeddings.shape[2:]],  # type: ignore
    )

    @tf_function
    def pseduo_interpolate(x):
        x = tf.broadcast_to(x, (num_steps, *x.shape))
        x = tf.reshape(x, (-1, *x.shape[2:]))
        return x

    interpolated_kwargs = tf.nest.map_structure(pseduo_interpolate, kwargs)
    interpolated_y_batch = pseduo_interpolate(y_batch)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_embeddings)
        logits = model(interpolated_embeddings, **interpolated_kwargs)
        logits_for_label = get_logits_for_labels(logits, interpolated_y_batch)

    grads = tape.gradient(logits_for_label, interpolated_embeddings)
    grads = tf.reshape(grads, [len(y_batch), num_steps, *grads.shape[1:]])
    return tfp.math.trapz(tfp.math.trapz(grads, axis=1), axis=-1)


def _integrated_gradients_iterative(
    model: TensorFlowTextClassifier,
    interpolated_embeddings_batch: List[tf.Tensor],
    y_batch: np.ndarray,
    **kwargs,
) -> np.ndarray:
    scores = []

    for i, interpolated_embeddings in enumerate(interpolated_embeddings_batch):
        interpolated_embeddings = tf.convert_to_tensor(interpolated_embeddings)

        interpolated_kwargs = map_dict(
            {k: v[i] for k, v in kwargs.items()},
            lambda x: tf.broadcast_to(x, (interpolated_embeddings.shape[0], *x.shape)),
        )
        with tf.GradientTape() as tape:
            tape.watch(interpolated_embeddings)
            logits = model(interpolated_embeddings, **interpolated_kwargs)
            logits_for_label = tf.gather(logits, axis=-1, indices=y_batch[i])

        grads = tape.gradient(logits_for_label, interpolated_embeddings)
        score = tfp.math.trapz(tfp.math.trapz(grads, axis=0), axis=-1)
        scores.append(score)

    return tf.stack(scores)


# ----------------------- NoiseGrad -------------------------


def _get_noise_grad_baseline_explain_fn(explain_fn: Callable | str) -> Callable:
    if isinstance(explain_fn, Callable):
        return explain_fn

    if explain_fn in ("NoiseGrad", "NoiseGrad++"):
        raise ValueError(f"Can't use {explain_fn} as baseline function for NoiseGrad.")
    method_mapping = available_explanation_functions()
    if explain_fn not in method_mapping:
        raise ValueError(
            f"Unknown XAI method {explain_fn}, supported are {list(method_mapping.keys())}"
        )
    return method_mapping[explain_fn]


@sd_explain_noise_grad.register
def _(
    x_batch: list,
    model: TensorFlowTextClassifier,
    y_batch: np.ndarray,
    *,
    mean: float = 1.0,
    std: float = 0.2,
    n: int = 10,
    explain_fn: Union[Callable, str] = "IntGrad",
    noise_type: str = "multiplicative",
    seed: int = 42,
    **kwargs,
) -> List[Explanation]:
    input_ids, _ = get_input_ids(x_batch, model)
    embeddings, kwargs = get_embeddings(x_batch, model)
    scores = sd_explain_noise_grad(
        embeddings,
        model,
        y_batch,
        mean=mean,
        std=std,
        n=n,
        explain_fn=explain_fn,
        noise_type=noise_type,
        seed=seed,
        **kwargs,
    )

    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@sd_explain_noise_grad.register
def _(
    x_batch: tf.Tensor,
    model: TensorFlowTextClassifier,
    y_batch: np.ndarray,
    *,
    n: int = 10,
    mean: float = 1.0,
    std: float = 0.2,
    noise_type: str = "multiplicative",
    explain_fn: Union[Callable, str] = "IntGrad",
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """A version of NoiseGrad++ explainer meant for usage with latent space perturbation."""

    explain_fn = _get_noise_grad_baseline_explain_fn(explain_fn)

    tf.random.set_seed(seed)
    original_weights = model.weights
    batch_size = x_batch.shape[0]
    num_tokens = x_batch.shape[1]

    explanations = []

    for _n in range(n):
        weights_copy = original_weights.copy()
        for index, params in enumerate(weights_copy):
            params_noise = tf.random.normal(params.shape, mean=mean, stddev=std)
            noisy_params = apply_noise(params, params_noise, noise_type)
            weights_copy[index] = noisy_params

        model.weights = weights_copy

        explanation = explain_fn(model, x_batch, y_batch, **kwargs)  # type: ignore
        explanations.append(explanation)

    scores = tf.reduce_mean(tf.stack(explanations), axis=0)

    model.weights = original_weights
    return scores


# ----------------------- NoiseGrad++ -------------------------


@sd_explain_noise_grad_plus_plus.register
def _(
    x_batch: list,
    model: TensorFlowTextClassifier,
    y_batch: np.ndarray,
    *,
    mean: float = 1.0,
    std: float = 0.2,
    sg_mean: float = 0.0,
    sg_std: float = 0.4,
    n: int = 10,
    m: int = 10,
    explain_fn: Union[Callable, str] = "IntGrad",
    noise_type: str = "multiplicative",
    seed: int = 42,
    **kwargs,
) -> List[Explanation]:
    input_ids, _ = get_input_ids(x_batch, model)
    embeddings, kwargs = get_embeddings(x_batch, model)
    scores = sd_explain_noise_grad_plus_plus(
        embeddings,
        model,
        y_batch,
        mean=mean,
        std=std,
        sg_mean=sg_mean,
        sg_std=sg_std,
        n=n,
        m=m,
        explain_fn=explain_fn,
        noise_type=noise_type,
        seed=seed,
        **kwargs,
    )

    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@sd_explain_noise_grad_plus_plus.register
def _(
    x_batch: tf.Tensor,
    model: TensorFlowTextClassifier,
    y_batch: np.ndarray,
    *,
    n: int = 10,
    m: int = 10,
    mean: float = 1.0,
    std: float = 0.2,
    sg_mean: float = 0.0,
    sg_std: float = 0.4,
    noise_type: str = "multiplicative",
    explain_fn: Union[Callable, str] = "IntGrad",
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """A version of NoiseGrad++ explainer meant for usage with latent space perturbation."""

    if isinstance(explain_fn, str):
        method_mapping = available_explanation_functions()

        if explain_fn == "NoiseGrad++":
            raise ValueError(
                "Can't use NoiseGrad++ as baseline explanation function for NoiseGrad++"
            )
        if explain_fn not in method_mapping:
            raise ValueError(
                f"Unsupported explain_fn {explain_fn}, supported are {list(method_mapping.keys())}"
            )
        explain_fn = method_mapping[explain_fn]

    tf.random.set_seed(seed)
    original_weights = model.weights
    batch_size = x_batch.shape[0]
    num_tokens = x_batch.shape[1]

    explanations = []

    for _n in range(n):
        noisy_explanations = []

        weights_copy = original_weights.copy()
        for index, params in enumerate(weights_copy):
            params_noise = tf.random.normal(params.shape, mean=mean, stddev=std)
            noisy_params = apply_noise(params, params_noise, noise_type)
            weights_copy[index] = noisy_params

        model.weights = weights_copy
        for _m in range(m):
            inputs_noise = tf.random.normal(x_batch.shape, mean=sg_mean, stddev=sg_std)
            noisy_embeddings = apply_noise(x_batch, inputs_noise, noise_type)
            explanation = explain_fn(model, noisy_embeddings, y_batch, **kwargs)  # type: ignore
            noisy_explanations.append(explanation)

        explanations.append(noisy_explanations)

    scores = tf.reduce_mean(explanations, axis=(0, 1))

    model.weights = original_weights

    return scores


# We need numpy to be defined during module load, but we must not use during runtime
del np
