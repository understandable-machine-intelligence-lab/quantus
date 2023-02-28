from __future__ import annotations

from typing import List, Optional, Callable, Dict, no_type_check

import numpy as np
from quantus.helpers.utils import calculate_auc

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.types import Explanation, NormaliseFn

from quantus.nlp.metrics.batched_metric import BatchedMetric
from quantus.nlp.functions.normalise_func import normalize_sum_to_1
from quantus.nlp.helpers.utils import safe_as_array, get_input_ids
from quantus.nlp.helpers.plotting import plot_token_flipping_experiment


class TokenFlipping(BatchedMetric):
    """
    References:
        - https://arxiv.org/abs/2202.07304
        - https://github.com/AmeenAli/XAI_Transformers/blob/main/SST/run_sst.py#L127
        - https://arxiv.org/pdf/2205.15389.pdf

    For the NLP experiments, we consider token sequences instead of graph nodes. The activation task starts with an
    empty sentence of “UNK” tokens, which are then gradually replaced with the original tokens in the order of highest
    to lowest relevancy. In the pruning task, we remove tokens from lowest to highest absolute relevance by replacing
    them with “UNK” tokens, similarly to the ablation experiments of Abnar & Zuidema (2020).
    """

    def __init__(
        self,
        *,
        abs: bool = False,
        normalise: bool = False,
        normalise_func: Optional[NormaliseFn] = normalize_sum_to_1,
        normalise_func_kwargs: Optional[Dict] = None,
        return_aggregate: bool = False,
        aggregate_func: Optional[Callable] = np.mean,
        disable_warnings: bool = False,
        display_progressbar: bool = False,
        return_auc_per_sample: bool = False,
        mask_token: str = "[UNK]",
        task: str = "pruning",
        default_plot_func: Optional[Callable] = plot_token_flipping_experiment,
    ):
        super().__init__(
            abs=abs,
            normalise=normalise,
            normalise_func=normalise_func,
            normalise_func_kwargs=normalise_func_kwargs,
            return_aggregate=return_aggregate,
            aggregate_func=aggregate_func,
            disable_warnings=disable_warnings,
            display_progressbar=display_progressbar,
            default_plot_func=default_plot_func,
        )
        self.return_auc_per_sample = return_auc_per_sample
        self.mask_token = mask_token
        if task not in ("pruning", "activation"):
            raise ValueError(
                f"Unknown value for {task}, allowed are: pruning, activation"
            )
        self.task = task

    @no_type_check
    def evaluate_batch(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
        *args,
        **kwargs,
    ) -> np.ndarray | float | Dict[str, float]:
        scores = np.asarray([])
        if self.task == "pruning":
            scores = self._eval_pruning(model, x_batch, y_batch, a_batch)
        elif self.task == "activation":
            scores = self._eval_activation(model, x_batch, y_batch, a_batch)

        # Move batch axis to 0's position.
        scores = scores.T

        if self.return_auc_per_sample:
            scores = calculate_auc(scores)

        return scores

    def _eval_activation(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
    ):
        batch_size = len(a_batch)
        num_tokens = len(a_batch[0][1])

        # We need to have tokens axis at positions 0, so we can just insert batches.
        scores = np.full(shape=(num_tokens, batch_size), fill_value=np.NINF)
        mask_indices_all = []
        for a in a_batch:
            # ascending
            token_indices_sorted_by_scores = np.argsort(a[1])[::-1]
            mask_indices_all.append(token_indices_sorted_by_scores)

        mask_indices_all = np.asarray(mask_indices_all).T
        input_ids, predict_kwargs = get_input_ids(x_batch, model)
        mask_token_id = model.tokenizer.token_id(self.mask_token)
        masked_input_ids = np.full_like(np.asarray(input_ids), fill_value=mask_token_id)

        for i, mask_indices_batch in enumerate(mask_indices_all):
            for index_in_batch, mask_index in enumerate(mask_indices_batch):
                masked_input_ids[index_in_batch][mask_index] = input_ids[
                    index_in_batch
                ][mask_index]

            embeddings = model.embedding_lookup(masked_input_ids)
            logits = model(embeddings, **predict_kwargs)
            logits = safe_as_array(logits)
            scores[i] = np.take(logits, y_batch)

        return scores

    def _eval_pruning(
        self,
        model: TextClassifier,
        x_batch: List[str],
        y_batch: np.ndarray,
        a_batch: List[Explanation],
    ):
        batch_size = len(a_batch)
        num_tokens = len(a_batch[0][1])

        # We need to have tokens axis at positions 0, so we can just insert batches.
        scores = np.full(shape=(num_tokens, batch_size), fill_value=np.NINF)
        mask_indices_all = []
        for a in a_batch:
            # descending
            token_indices_sorted_by_scores = np.argsort(a[1])
            mask_indices_all.append(token_indices_sorted_by_scores)

        mask_indices_all = np.asarray(mask_indices_all).T
        input_ids, predict_kwargs = get_input_ids(x_batch, model)
        mask_token_id = model.tokenizer.token_id(self.mask_token)

        for i, mask_indices_batch in enumerate(mask_indices_all):
            for index_in_batch, mask_index in enumerate(mask_indices_batch):
                input_ids[index_in_batch][mask_index] = mask_token_id

            embeddings = model.embedding_lookup(input_ids)
            logits = model(embeddings, **predict_kwargs)
            logits = safe_as_array(logits)
            scores[i] = np.take(logits, y_batch)

        return scores
