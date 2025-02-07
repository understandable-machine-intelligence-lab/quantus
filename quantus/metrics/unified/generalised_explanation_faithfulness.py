"""This module contains the implementation of the Completeness metric."""

# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from quantus.functions.perturb_func import baseline_replacement_by_indices
from quantus.helpers import warn
from quantus.helpers.enums import (
    DataType,
    EvaluationCategory,
    ModelType,
    ScoreDirection,
)
from quantus.helpers.model.model_interface import ModelInterface
from quantus.helpers.perturbation_utils import make_perturb_func
from quantus.metrics.base import Metric
from quantus.helpers.utils import identity

if sys.version_info >= (3, 8):
    from typing import final
else:
    from typing_extensions import final


@final
class GeneralisedExplanationFaithfulness(Metric[List[float]]):
    """
    Implementation of Generalised Explanation Faithfulness test by Hedström et al., 2025.

    Insert desription

    References:
        1) Hedström et al., "Evaluating Interpretable Methods via Geometric Alignment of Functional Distortions"
       Transactions of Machine Learning Research, 2025.

    Attributes:
        -  _name: The name of the metric.
        - _data_applicability: The data types that the metric implementation currently supports.
        - _models: The model types that this metric can work with.
        - score_direction: How to interpret the scores, whether higher/ lower values are considered better.
    """

    name = "GEF"
    data_applicability = {DataType.IMAGE, DataType.TIMESERIES, DataType.TABULAR}
    model_applicability = {ModelType.TORCH}
    score_direction = ScoreDirection.HIGHER
    evaluation_category = EvaluationCategory.UNIFIED

    def __init__()
        pass