from __future__ import annotations

from importlib import util

from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.functions.perturb_func import (
    gaussian_noise,
    spelling_replacement,
    synonym_replacement,
    typo_replacement,
    uniform_noise,
)
from quantus.nlp.helpers.constants import (
    available_categories,
    available_latent_space_perturbation_functions,
    available_metrics,
    available_metrics_numerical_perturbation,
    available_metrics_plain_text_perturbation,
    available_normalisation_functions,
    available_numerical_xai_methods,
    available_perturbation_functions,
    available_plain_text_perturbation_functions,
    available_plain_text_xai_methods,
    available_xai_methods,
)
from quantus.nlp.helpers.model.huggingface_tokenizer import HuggingFaceTokenizer
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.plotting import (
    visualise_explanations_as_html,
    visualise_explanations_as_pyplot,
)

if util.find_spec("tensorflow"):
    from quantus.nlp.helpers.model.tensorflow_text_classifier import (
        TensorFlowTextClassifier,
    )

    if util.find_spec("transformers"):
        from quantus.nlp.helpers.model.tensorflow_huggingface_text_classifier import (
            TensorFlowHuggingFaceTextClassifier,
        )

if util.find_spec("torch"):
    from quantus.nlp.helpers.model.torch_text_classifier import TorchTextClassifier

    if util.find_spec("transformers"):
        from quantus.nlp.helpers.model.torch_huggingface_text_classifier import (
            TorchHuggingFaceTextClassifier,
        )

from quantus.nlp.evaluation import evaluate
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.plotting import plot_token_flipping_experiment
from quantus.nlp.helpers.types import (
    ExplainFn,
    Explanation,
    NormaliseFn,
    PerturbFn,
    SimilarityFn,
)
from quantus.nlp.metrics.faithfullness.token_flipping import TokenFlipping
from quantus.nlp.metrics.randomisation.model_parameter_randomisation import (
    ModelParameterRandomisation,
)
from quantus.nlp.metrics.randomisation.random_logit import RandomLogit
from quantus.nlp.metrics.robustness.avg_sensitivity import AvgSensitivity
from quantus.nlp.metrics.robustness.max_sensitivity import MaxSensitivity
from quantus.nlp.metrics.robustness.relative_input_stability import (
    RelativeInputStability,
)
from quantus.nlp.metrics.robustness.relative_output_stability import (
    RelativeOutputStability,
)
from quantus.nlp.metrics.robustness.relative_representation_stability import (
    RelativeRepresentationStability,
)
