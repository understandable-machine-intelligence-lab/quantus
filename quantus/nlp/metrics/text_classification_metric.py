from __future__ import annotations

from abc import abstractmethod
from functools import partial
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from quantus.nlp.functions.explanation_func import explain
from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import ExplainFn, Explanation
from quantus.nlp.helpers.utils import (
    batch_list,
    map_explanations,
    map_optional,
    value_or_default,
)


class TextClassificationMetric:
    def __init__(
        self,
        abs: bool,  # noqa
        normalise: bool,
        normalise_func: Optional[Callable],
        normalise_func_kwargs: Optional[Dict],
        return_aggregate: bool,
        aggregate_func: Optional[Callable],
        disable_warnings: bool,
        display_progressbar: bool,
    ):
        self.abs = abs
        self.normalise = normalise
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = value_or_default(normalise_func_kwargs, lambda: {})
        self.return_aggregate = return_aggregate
        self.aggregate_func = aggregate_func
        self.disable_warnings = disable_warnings
        self.display_progressbar = display_progressbar

    def __call__(
        self,
        model: TextClassifier,
        x_batch: List[str],
        *,
        y_batch: Optional[np.ndarray] = None,
        a_batch: Optional[List[Explanation] | np.ndarray] = None,
        explain_func: ExplainFn = explain,
        explain_func_kwargs: Optional[Dict] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        explain_func_kwargs = value_or_default(explain_func_kwargs, lambda: {})
        x_batch = batch_list(x_batch, batch_size)  # type: ignore
        y_batch = map_optional(y_batch, partial(batch_list, batch_size=batch_size))
        a_batch = map_optional(a_batch, partial(batch_list, batch_size=batch_size))
        pbar = tqdm(
            x_batch,
            disable=not self.display_progressbar,
            total=len(x_batch),
        )

        scores_batch = []

        for i, x in enumerate(pbar):
            y = map_optional(y_batch, itemgetter(i))
            a = map_optional(a_batch, itemgetter(i))
            x, y, a, custom = self.batch_preprocess(
                model, x, y, a, explain_func, explain_func_kwargs
            )
            score = self.evaluate_batch(
                model,
                x,
                y,
                a,
                explain_func=explain_func,
                explain_func_kwargs=explain_func_kwargs,
                custom_batch=custom,
            )
            score = self.batch_postprocess(
                model, x, y, a, explain_func, explain_func_kwargs, score
            )
            scores_batch.append(score)

        return np.reshape(scores_batch, -1)

    def explain_batch(
        self,
        model: TextClassifier,
        x_batch: List[str] | np.ndarray,
        y_batch: np.ndarray,
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
    ) -> List[Explanation] | np.ndarray:
        a_batch = explain_func(model, x_batch, y_batch, **explain_func_kwargs)  # type: ignore

        if self.normalise:
            a_batch = map_explanations(
                a_batch, partial(self.normalise_func, **self.normalise_func_kwargs)
            )

        if self.abs:
            a_batch = map_explanations(a_batch, np.abs)

        return a_batch

    def batch_preprocess(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: Optional[np.ndarray],
        a_batch: Optional[List[Explanation]],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
    ) -> Tuple[List[str], np.ndarray, List[Explanation], Optional[None]]:
        y_batch = value_or_default(
            y_batch, lambda: model.predict(x_batch).argmax(axis=-1)
        )
        a_batch = value_or_default(
            a_batch,
            lambda: self.explain_batch(
                model, x_batch, y_batch, explain_func, explain_func_kwargs
            ),
        )
        return x_batch, y_batch, a_batch, None

    def batch_postprocess(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: Optional[np.ndarray],
        a_batch: Optional[List[Explanation]],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
        score: np.ndarray,
    ) -> np.ndarray:
        return score

    @abstractmethod
    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: Optional[np.ndarray],
        a_batch: Optional[List[Explanation]],
        explain_func: ExplainFn,
        explain_func_kwargs: Dict,
        custom_batch: Optional[Any] = None,
    ) -> np.ndarray:
        """Must be implemented by respective metric class."""
        raise NotImplementedError  # pragma: not covered
