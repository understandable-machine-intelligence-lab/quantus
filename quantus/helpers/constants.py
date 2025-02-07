"""This module contains constants and simple methods to retreive the available metrics, perturbation-,
similarity-, normalisation- functions and explanation methods in Quantus."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import sys
from typing import List, Dict, Mapping, Type
from quantus.functions.loss_func import *
from quantus.functions.normalise_func import *
from quantus.functions.perturb_func import *
from quantus.functions.similarity_func import *
from quantus.functions import n_bins_func
from quantus.metrics import *

if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final


AVAILABLE_METRICS: Final[Mapping[str, Mapping[str, Type[Metric]]]] = {
    "Faithfulness": {
        "Faithfulness Correlation": FaithfulnessCorrelation,
        "Faithfulness Estimate": FaithfulnessEstimate,
        "Pixel-Flipping": PixelFlipping,
        "Region Segmentation": RegionPerturbation,
        "Monotonicity-Arya": Monotonicity,
        "Monotonicity-Nguyen": MonotonicityCorrelation,
        "Selectivity": Selectivity,
        "SensitivityN": SensitivityN,
        "IROF": IROF,
        "ROAD": ROAD,
        "Infidelity": Infidelity,
        "Sufficiency": Sufficiency,
    },
    "Robustness": {
        "Continuity Test": Continuity,
        "Local Lipschitz Estimate": LocalLipschitzEstimate,
        "Max-Sensitivity": MaxSensitivity,
        "Avg-Sensitivity": AvgSensitivity,
        "Consistency": Consistency,
        "Relative Input Stability": RelativeInputStability,
        "Relative Output Stability": RelativeOutputStability,
        "Relative Representation Stability": RelativeRepresentationStability,
    },
    "Localisation": {
        "Pointing Game": PointingGame,
        "Top-K Intersection": TopKIntersection,
        "Relevance Mass Accuracy": RelevanceMassAccuracy,
        "Relevance Rank Accuracy": RelevanceRankAccuracy,
        "Attribution Localisation ": AttributionLocalisation,
        "AUC": AUC,
        "Focus": Focus,
    },
    "Complexity": {
        "Sparseness": Sparseness,
        "Complexity": Complexity,
        "Effective Complexity": EffectiveComplexity,
    },
    "Randomisation": {
        "MPRT": MPRT,
        "Smooth MPRT": SmoothMPRT,
        "Efficient MPRT": EfficientMPRT,
        "Random Logit": RandomLogit,
    },
    "Axiomatic": {
        "Completeness": Completeness,
        "NonSensitivity": NonSensitivity,
        "InputInvariance": InputInvariance,
    },
    "Unified": {
        "Generalised Explanation Faithfulness": GeneralisedExplanationFaithfulness
        }
}


AVAILABLE_PERTURBATION_FUNCTIONS = {
    "baseline_replacement_by_indices": baseline_replacement_by_indices,
    "baseline_replacement_by_shift": baseline_replacement_by_shift,
    "baseline_replacement_by_blur": baseline_replacement_by_blur,
    "gaussian_noise": gaussian_noise,
    "uniform_noise": uniform_noise,
    "rotation": rotation,
    "translation_x_direction": translation_x_direction,
    "translation_y_direction": translation_y_direction,
    "no_perturbation": no_perturbation,
    "noisy_linear_imputation": noisy_linear_imputation,
}


AVAILABLE_SIMILARITY_FUNCTIONS = {
    "correlation_spearman": correlation_spearman,
    "correlation_pearson": correlation_pearson,
    "correlation_kendall_tau": correlation_kendall_tau,
    "distance_euclidean": distance_euclidean,
    "distance_manhattan": distance_manhattan,
    "distance_chebyshev": distance_chebyshev,
    "lipschitz_constant": lipschitz_constant,
    "abs_difference": abs_difference,
    "squared_difference": squared_difference,
    "difference": difference,
    "cosine": cosine,
    "ssim": ssim,
    "mse": mse,
}

AVAILABLE_NORMALISATION_FUNCTIONS = {
    "normalise_by_negative": normalise_by_negative,
    "normalise_by_max": normalise_by_max,
    "denormalise": denormalise,
}


AVAILABLE_XAI_METHODS_CAPTUM = [
    "GradientShap",
    "IntegratedGradients",
    "DeepLift",
    "DeepLiftShap",
    "InputXGradient",
    "Saliency",
    "FeatureAblation",
    "Deconvolution",
    "FeaturePermutation",
    "Lime",
    "KernelShap",
    "LRP",
    "Gradient",
    "Occlusion",
    "LayerGradCam",
    "GuidedGradCam",
    "LayerConductance",
    "LayerActivation",
    "InternalInfluence",
    "LayerGradientXActivation",
    "Control Var. Sobel Filter",
    "Control Var. Constant",
    "Control Var. Random Uniform",
]


DEPRECATED_XAI_METHODS_CAPTUM = {"GradCam": "LayerGradCam"}


AVAILABLE_XAI_METHODS_TF = [
    "VanillaGradients",
    "IntegratedGradients",
    "GradientsInput",
    "OcclusionSensitivity",
    "GradCAM",
    "SmoothGrad",
]


DEPRECATED_XAI_METHODS_TF = {
    "Gradient": "VanillaGradients",
    "InputXGradient": "GradientsInput",
    "Occlusion": "OcclusionSensitivity",
    "GradCam": "GradCAM",
}


AVAILABLE_N_BINS_ALGORITHMS = {
    "Freedman Diaconis": n_bins_func.freedman_diaconis_rule,
    "Scotts": n_bins_func.scotts_rule,
    "Square Root": n_bins_func.square_root_choice,
    "Sturges Formula": n_bins_func.sturges_formula,
    "Rice": n_bins_func.rice_rule,
}


def available_categories() -> List[str]:
    """
    Retrieve the available metric categories in Quantus.

    Returns
    -------
    List[str]
        With the available metric categories in Quantus.
    """
    return [c for c in AVAILABLE_METRICS.keys()]


def available_metrics() -> Dict[str, List[str]]:
    """
    Retrieve the available metrics in Quantus.

    Returns
    -------
    Dict[str, str]
        With the available metrics, under each category in Quantus.
    """
    return {c: list(metrics.keys()) for c, metrics in AVAILABLE_METRICS.items()}


def available_methods_tf_explain() -> List[str]:
    """
    Retrieve the available explanation methods in Quantus.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_XAI_METHODS_TF]


def available_methods_captum() -> List[str]:
    """
    Retrieve the available explanation methods in Quantus.

    Returns
    -------
    List[str]
        With the available explanation methods in Quantus.
    """
    return [c for c in AVAILABLE_XAI_METHODS_CAPTUM]


def available_perturbation_functions() -> List[str]:
    """
    Retrieve the available perturbation functions in Quantus.

    Returns
    -------
    List[str]
        With the available perturbation functions in Quantus.
    """
    return [c for c in AVAILABLE_PERTURBATION_FUNCTIONS.keys()]


def available_similarity_functions() -> List[str]:
    """
    Retrieve the available similarity functions in Quantus.

    Returns
    -------
    List[str]
        With the available similarity functions in Quantus.
    """
    return [c for c in AVAILABLE_SIMILARITY_FUNCTIONS.keys()]


def available_normalisation_functions() -> List[str]:
    """
    Retrieve the available normalisation functions in Quantus.

    Returns
    -------
    List[str]
        With the available normalisation functions in Quantus.
    """
    return [c for c in AVAILABLE_NORMALISATION_FUNCTIONS.keys()]
