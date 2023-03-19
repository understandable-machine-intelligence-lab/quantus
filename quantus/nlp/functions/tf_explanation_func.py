# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

"""Explanation functions for TensorFlow models."""
from __future__ import annotations

from functools import partial, singledispatch
from typing import Callable, Dict, List, Optional, Union, NamedTuple

import numpy
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions.normal import Normal
from quantus.nlp.helpers.model.tf_model import TFHuggingFaceTextClassifier
from quantus.nlp.helpers.types import Explanation
from quantus.nlp.helpers.utils import (
    get_input_ids,
    tf_function,
    value_or_default,
    apply_noise,
)

# Just to save some typing effort
_BaselineFn = Callable[[tf.Tensor], tf.Tensor]
_TextOrVector = Union[List[str], tf.Tensor]
_Scores = Union[List[Explanation], tf.Tensor]

# ----------------- "Entry Point" --------------------


class IntGradConfig(NamedTuple):
    """
    num_steps:
        Number of interpolated samples, which should be generated, default=10.
    baseline_fn:
        Function used to created baseline values, by default will create zeros tensor. Alternatively, e.g.,
        embedding for [UNK] token could be used.
    batch_interpolated_inputs:
        Indicates if interpolated inputs should be stacked into 1 bigger batch.
        This speeds up the explanation, however can be very memory intensive.
    """

    num_steps: int = 10
    baseline_fn: _BaselineFn = tf_function(lambda x: tf.zeros_like(x, dtype=x.dtype))
    batch_interpolated_inputs: bool = True


class NoiseGradConfig(NamedTuple):
    """
    mean:
        Mean of normal distribution, from which noise applied to model's weights is sampled, default=1.0.
    std:
        Standard deviation of normal distribution, from which noise applied to model's weights is sampled, default=0.2.
    n:
        Number of times noise is applied to weights, default=10.
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad, default=IntGrad.
        Passing additional kwargs is not supported, please use partial application from functools package instead.

    noise_type:
        If multiplicative weights and input embeddings will be multiplied by noise.
        If additive noise will be added to weights and input embeddings.

    seed:
        PRNG seed used for noise generating distributions.
    """

    n: int = 10
    mean: float = 1.0
    std: float = 0.2
    explain_fn: Union[Callable, str] = "IntGrad"
    noise_type: str = "multiplicative"
    seed: int = 42

    def resolve_functions(self):
        explain_fn = _resolve_noise_grad_baseline_explain_fn(self.explain_fn)
        return NoiseGradConfig(
            n=self.n,
            mean=self.mean,
            std=self.std,
            explain_fn=explain_fn,
            noise_type=self.noise_type,
            seed=self.seed,
        )


class NoiseGradPlusPlusConfig(NamedTuple):
    """
    mean:
        Mean of normal distribution, from which noise applied to model's weights is sampled, default=1.0.
    sg_mean:
        Mean of normal distribution, from which noise applied to input embeddings is sampled, default=0.0.
    std:
        Standard deviation of normal distribution, from which noise applied to model's weights is sampled, default=0.2.
    sg_std:
        Standard deviation of normal distribution, from which noise applied to input embeddings is sampled, default=0.4.
    n:
        Number of times noise is applied to weights, default=10.
      m:
        Number of times noise is applied to input embeddings, default=10
    explain_fn:
        Baseline explanation function. If string provided must be one of GradNorm, GradXInput, IntGrad, default=IntGrad.
        Passing additional kwargs is not supported, please use partial application from functools package instead.
    noise_type:
        If multiplicative weights and input embeddings will be multiplied by noise.
        If additive noise will be added to weights and input embeddings.

    seed:
        PRNG seed used for noise generating distributions.
    """

    n: int = 10
    m: int = 10
    mean: float = 1.0
    sg_mean: float = 0.0
    std: float = 0.2
    sg_std: float = 0.4
    explain_fn: Union[Callable, str] = "IntGrad"
    noise_type: str = "multiplicative"
    seed: int = 42

    def resolve_functions(self):
        explain_fn = _resolve_noise_grad_baseline_explain_fn(self.explain_fn)
        return NoiseGradPlusPlusConfig(
            n=self.n,
            m=self.m,
            mean=self.mean,
            sg_mean=self.sg_mean,
            std=self.std,
            sg_std=self.sg_std,
            explain_fn=explain_fn,
            noise_type=self.noise_type,
            seed=self.seed,
        )


def available_xai_methods() -> Dict[str, Callable]:
    return {
        "GradNorm": gradient_norm,
        "GradXInput": gradient_x_input,
        "IntGrad": integrated_gradients,
        "NoiseGrad": noise_grad,
        "NoiseGrad++": noise_grad_plus_plus,
    }


def available_noise_grad_xai_methods() -> Dict[str, Callable]:
    return {
        "GradNorm": _gradient_norm,
        "GradXInput": _gradient_x_input,
        "IntGrad": _integrated_gradients,
    }


def tf_explain(
    *args,
    method: str,
    **kwargs,
) -> _Scores:
    """Execute gradient based explanation method."""

    method_mapping = available_xai_methods()

    if method not in method_mapping:
        raise ValueError(
            f"Unsupported explanation method: {method}, supported are: {list(method_mapping.keys())}"
        )
    explain_fn = method_mapping[method]

    def as_tensor(x):
        if isinstance(x, numpy.ndarray):
            return tf.constant(x)
        else:
            return x

    args = tf.nest.map_structure(as_tensor, args)
    kwargs = tf.nest.map_structure(as_tensor, kwargs)

    return explain_fn(*args, **kwargs)


# ----------------- Quantus-conform API -------------------
# functools.singledispatch supports only dispatching based on 1st argument type,
# which in our case is model, so we need to reorder them, so x_batch (text or embedding) is in 1st place,
# and we and up dispatching to different functions based on input type.


def gradient_norm(
    model: TFHuggingFaceTextClassifier,
    x_batch: _TextOrVector,
    y_batch: tf.Tensor,
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
        A batch of plain text inputs or their embeddings, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    return _gradient_norm(x_batch, model, y_batch, **kwargs)


def gradient_x_input(
    model: TFHuggingFaceTextClassifier,
    x_batch: _TextOrVector,
    y_batch: tf.Tensor,
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
        A batch of plain text inputs or their embeddings, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    """
    return _gradient_x_input(x_batch, model, y_batch, **kwargs)


def integrated_gradients(
    model: TFHuggingFaceTextClassifier,
    x_batch: _TextOrVector,
    y_batch: tf.Tensor,
    config: Optional[IntGradConfig] = None,
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
        A batch of plain text inputs or their embeddings, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    config:

    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.

    Examples
    -------
    Specifying [UNK] token as baseline:

    >>> unk_token_embedding = model.embedding_lookup([model.unwrap_tokenizer().unk_token_id])[0, 0]
    >>> unknown_token_baseline_function = tf.function(lambda x: unk_token_embedding)
    >>> config = IntGradConfig(baseline_fn=unknown_token_baseline_function)
    >>> integrated_gradients(..., ..., ..., config=config)

    """
    config = value_or_default(config, lambda: IntGradConfig())
    return _integrated_gradients(
        x_batch,
        model,
        y_batch,
        config=config,
        **kwargs,
    )


def noise_grad(
    model: TFHuggingFaceTextClassifier,
    x_batch: _TextOrVector,
    y_batch: tf.Tensor,
    config: Optional[NoiseGradConfig] = None,
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
        A batch of plain text inputs or their embeddings, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    config:

    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.


    Examples
    -------
    Passing kwargs to baseline explanation function:

    >>> import functools
    >>> ig_config = IntGradConfig(num_steps=22)
    >>> explain_fn = functools.partial(integrated_gradients, config=ig_config)
    >>> ng_config = NoiseGradConfig(explain_fn=explain_fn)
    >>> noise_grad_plus_plus(config=ng_config)

    References
    -------
    - https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.
    - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185

    """

    config = value_or_default(config, lambda: NoiseGradConfig()).resolve_functions()
    tf.keras.utils.set_random_seed(config.seed)

    return _noise_grad(
        x_batch,
        model,
        y_batch,
        config=config,
        **kwargs,
    )


def noise_grad_plus_plus(
    model: TFHuggingFaceTextClassifier,
    x_batch: _TextOrVector,
    y_batch: tf.Tensor,
    config: Optional[NoiseGradPlusPlusConfig] = None,
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
        A batch of plain text inputs or their embeddings, which are subjects to explanation.
    y_batch:
        A batch of labels, which are subjects to explanation.
    config:

    kwargs:
        If x_batch is embeddings, kwargs can be used to pass, additional forward pass kwargs, e.g., attention mask.

    Returns
    -------
    a_batch:
        List of tuples, where 1st element is tokens and 2nd is the scores assigned to the tokens.


    Examples
    -------
    Passing kwargs to baseline explanation function:

    References
    -------
    - https://github.com/understandable-machine-intelligence-lab/NoiseGrad/blob/master/src/noisegrad.py#L80.
    - Kirill Bykov and Anna Hedström and Shinichi Nakajima and Marina M. -C. Höhne, 2021, NoiseGrad: enhancing explanations by introducing stochasticity to model weights, https://arxiv.org/abs/2106.10185

    """

    config = value_or_default(
        config, lambda: NoiseGradPlusPlusConfig()
    ).resolve_functions()
    tf.keras.utils.set_random_seed(config.seed)
    return _noise_grad_plus_plus(
        x_batch,
        model,
        y_batch,
        config=config,
        **kwargs,
    )


# --------- Single dispatch generic function stubs -----------


@singledispatch
def _gradient_norm(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def _gradient_x_input(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def _integrated_gradients(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def _noise_grad(x_batch, *args, **kwargs) -> _Scores:
    pass


@singledispatch
def _noise_grad_plus_plus(x_batch, *args, **kwargs) -> _Scores:
    pass


# ----------------------- GradNorm -------------------------


@_gradient_norm.register
def _(
    x_batch: list, model: TFHuggingFaceTextClassifier, y_batch: tf.Tensor
) -> List[Explanation]:
    input_ids, kwargs = get_input_ids(x_batch, model)
    embeddings = model.embedding_lookup(input_ids)
    scores = _gradient_norm(embeddings, model, y_batch, **kwargs)
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_gradient_norm.register(tf.Tensor)
@tf_function
def _(
    x_batch: tf.Tensor,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    **kwargs,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(x_batch, **kwargs)
        logits_for_label = _logits_for_labels(logits, y_batch)

    grads = tape.gradient(logits_for_label, x_batch)
    return tf.linalg.norm(grads, axis=-1)


# ----------------------- GradXInput -------------------------


@_gradient_x_input.register
def _(
    x_batch: list, model: TFHuggingFaceTextClassifier, y_batch: tf.Tensor, **kwargs
) -> List[Explanation]:
    input_ids, kwargs = get_input_ids(x_batch, model)
    embeddings = model.embedding_lookup(input_ids)
    scores = _gradient_x_input(embeddings, model, y_batch, **kwargs)
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_gradient_x_input.register(tf.Tensor)
@tf_function
def _(
    x_batch: tf.Tensor,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    **kwargs,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x_batch)
        logits = model(x_batch, **kwargs)
        logits_for_label = _logits_for_labels(logits, y_batch)

    grads = tape.gradient(logits_for_label, x_batch)
    return tf.math.reduce_sum(x_batch * grads, axis=-1)


# ----------------------- IntGrad ------------------------


@_integrated_gradients.register
def _(
    x_batch: list,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    config: IntGradConfig,
    **kwargs,
) -> List[Explanation]:
    input_ids, predict_kwargs = get_input_ids(x_batch, model)
    embeddings = model.embedding_lookup(input_ids)

    scores = _integrated_gradients(
        embeddings,
        model,
        y_batch,
        config,
        **kwargs,
        **predict_kwargs,
    )
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_integrated_gradients.register(tf.Tensor)
@tf_function
def _(
    x_batch: tf.Tensor,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    config: IntGradConfig,
    **kwargs,
):
    if config.batch_interpolated_inputs:
        return _integrated_gradients_batched(
            x_batch,
            model,
            y_batch,
            config,
            **kwargs,
        )
    else:
        return _integrated_gradients_iterative(
            x_batch,
            model,
            y_batch,
            config,
            **kwargs,
        )


@tf_function
def _integrated_gradients_batched(
    x_batch: tf.Tensor,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    config: IntGradConfig,
    **kwargs,
):
    num_steps = config.num_steps
    interpolated_embeddings = tf.vectorized_map(
        lambda i: _interpolate_inputs(config.baseline_fn(i), i, num_steps), x_batch
    )

    shape = tf.shape(interpolated_embeddings)
    batch_size = shape[0]

    interpolated_embeddings = tf.reshape(
        tf.cast(interpolated_embeddings, dtype=tf.float32),
        [-1, shape[2], shape[3]],
    )

    interpolated_kwargs = tf.nest.map_structure(
        partial(_pseudo_interpolate_batch, num_steps=num_steps), kwargs
    )
    interpolated_y_batch = _pseudo_interpolate_batch(y_batch, num_steps)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_embeddings)
        logits = model(interpolated_embeddings, **interpolated_kwargs)
        logits_for_label = _logits_for_labels(logits, interpolated_y_batch)

    grads = tape.gradient(logits_for_label, interpolated_embeddings)
    grads_shape = tf.shape(grads)
    grads = tf.reshape(
        grads, [batch_size, num_steps + 1, grads_shape[1], grads_shape[2]]
    )
    return tfp.math.trapz(tfp.math.trapz(grads, axis=1), axis=-1)


@tf_function
def _integrated_gradients_iterative(
    x_batch: tf.Tensor,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    config: IntGradConfig,
    **kwargs,
) -> tf.Tensor:
    interpolated_embeddings_batch = tf.map_fn(
        lambda x: _interpolate_inputs(config.baseline_fn(x), x, config.num_steps),
        x_batch,
    )

    batch_size = tf.shape(interpolated_embeddings_batch)[0]

    scores = tf.TensorArray(
        x_batch.dtype,
        size=batch_size,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    for i in tf.range(batch_size):
        interpolated_embeddings = interpolated_embeddings_batch[i]

        interpolated_kwargs = tf.nest.map_structure(
            lambda x: _pseudo_interpolate_single(x, interpolated_embeddings),
            {k: v[i] for k, v in kwargs.items()},
        )
        with tf.GradientTape() as tape:
            tape.watch(interpolated_embeddings)
            logits = model(interpolated_embeddings, **interpolated_kwargs)
            logits_for_label = logits[:, y_batch[i]]

        grads = tape.gradient(logits_for_label, interpolated_embeddings)
        score = tfp.math.trapz(tfp.math.trapz(grads, axis=0), axis=-1)
        scores = scores.write(i, score)

    return scores.stack()


# ----------------------- NoiseGrad -------------------------


@_noise_grad.register
def _(
    x_batch: list,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    config: NoiseGradConfig,
):
    input_ids, predict_kwargs = get_input_ids(x_batch, model)
    embeddings = model.embedding_lookup(input_ids)
    scores = _noise_grad(
        embeddings,
        model,
        y_batch,
        config,
        **predict_kwargs,
    )
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_noise_grad.register
def _(
    x_batch: tf.Tensor,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    config: NoiseGradConfig,
    **kwargs,
) -> tf.Tensor:
    original_weights = model.weights.copy()

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    noise_dist = Normal(config.mean, config.std)

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return apply_noise(x, noise, config.noise_type)

    for n in tf.range(config.n):
        noisy_weights = tf.nest.map_structure(
            noise_fn,
            original_weights,
        )
        model.weights = noisy_weights

        explanation = config.explain_fn(x_batch, model, y_batch, **kwargs)  # type: ignore # noqa
        explanations_array = explanations_array.write(n, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    model.weights = original_weights
    return scores


# ----------------------- NoiseGrad++ -------------------------


@_noise_grad_plus_plus.register
def _(
    x_batch: list,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    *,
    config: NoiseGradPlusPlusConfig,
):
    input_ids, kwargs = get_input_ids(x_batch, model)
    embeddings = model.embedding_lookup(input_ids)
    scores = _noise_grad_plus_plus(
        embeddings,
        model,
        y_batch,
        config=config,
        **kwargs,
    )
    return [(model.convert_ids_to_tokens(i), j) for i, j in zip(input_ids, scores)]


@_noise_grad_plus_plus.register
def _(
    x_batch: tf.Tensor,
    model: TFHuggingFaceTextClassifier,
    y_batch: tf.Tensor,
    *,
    config: NoiseGradPlusPlusConfig,
    **kwargs,
) -> tf.Tensor:
    original_weights = model.weights.copy()

    noise_dist = Normal(config.mean, config.std)
    sg_noise_dist = Normal(config.sg_mean, config.sg_std)

    explanations_array = tf.TensorArray(
        x_batch.dtype,
        size=config.n * config.m,
        clear_after_read=True,
        colocate_with_first_write_call=True,
    )

    def noise_fn(x):
        noise = noise_dist.sample(tf.shape(x))
        return apply_noise(x, noise, config.noise_type)

    def sg_noise_fn(x):
        noise = sg_noise_dist.sample(tf.shape(x))
        return apply_noise(x, noise, config.noise_type)

    for n in tf.range(config.n):
        noisy_weights = tf.nest.map_structure(noise_fn, original_weights)
        model.weights = noisy_weights

        for m in tf.range(config.m):
            noisy_embeddings = sg_noise_fn(x_batch)
            explanation = config.explain_fn(noisy_embeddings, model, y_batch, **kwargs)  # type: ignore # noqa
            explanations_array = explanations_array.write(n + m * config.m, explanation)

    scores = tf.reduce_mean(explanations_array.stack(), axis=0)
    model.weights = original_weights
    return scores


# --------------------- utils ----------------------


@tf_function
def _logits_for_labels(logits: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
    # Matrix with indexes like [ [0,y_0], [1, y_1], ...]
    indexes = tf.transpose(
        tf.stack(
            [
                tf.range(tf.shape(logits)[0], dtype=tf.int32),
                tf.cast(y_batch, tf.int32),
            ]
        ),
        [1, 0],
    )
    return tf.gather_nd(logits, indexes)


@tf_function
def _interpolate_inputs(
    baseline: tf.Tensor, target: tf.Tensor, num_steps: int
) -> tf.Tensor:
    """Gets num_step linearly interpolated inputs from baseline to target."""
    delta = target - baseline
    scales = tf.linspace(0, 1, num_steps + 1)[:, tf.newaxis, tf.newaxis]
    scales = tf.cast(scales, dtype=delta.dtype)
    shape = tf.convert_to_tensor(
        [num_steps + 1, tf.shape(delta)[0], tf.shape(delta)[1]]
    )
    deltas = scales * tf.broadcast_to(delta, shape)
    interpolated_inputs = baseline + deltas
    return interpolated_inputs


@tf_function
def _pseudo_interpolate_batch(x, num_steps):
    og_shape = tf.convert_to_tensor(tf.shape(x))
    new_shape = tf.concat([tf.constant([num_steps + 1]), og_shape], axis=0)
    x = tf.broadcast_to(x, new_shape)
    flat_shape = tf.concat([tf.constant([-1]), og_shape[1:]], axis=0)
    x = tf.reshape(x, flat_shape)
    return x


@tf_function
def _pseudo_interpolate_single(x, interpolated_embeddings):
    return tf.broadcast_to(x, (tf.shape(interpolated_embeddings)[0], *x.shape))


# -------------- not compiled functions ----------


def _resolve_noise_grad_baseline_explain_fn(explain_fn):
    if isinstance(explain_fn, Callable):
        return explain_fn  # type: ignore

    method_mapping = available_noise_grad_xai_methods()
    if explain_fn not in method_mapping:
        raise ValueError(
            f"Unknown XAI method {explain_fn}, supported are {list(method_mapping.keys())}"
        )
    return method_mapping[explain_fn]
