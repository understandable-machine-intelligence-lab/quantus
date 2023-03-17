# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
# Quantus project URL: <https://github.com/understandable-machine-intelligence-lab/Quantus>.

from __future__ import annotations

from abc import ABC
from typing import Dict, List

import numpy as np
from transformers import PreTrainedTokenizerBase

from quantus.nlp.helpers.model.text_classifier import TextClassifier
from quantus.nlp.helpers.utils import add_default_items


class HuggingFaceTokenizer(TextClassifier, ABC):
    """A wrapper around HuggingFace's hub tokenizers, which encapsulates common functionality used in Quantus."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self._tokenizer = tokenizer

    def batch_encode(self, text: List[str], **kwargs) -> Dict[str, np.ndarray]:  # type: ignore
        kwargs = add_default_items(kwargs, {"padding": "longest"})
        return self._tokenizer(text, **kwargs).data

    def convert_ids_to_tokens(self, ids: np.ndarray) -> List[str]:
        return self._tokenizer.convert_ids_to_tokens(ids)

    def token_id(self, token: str) -> int:
        return self._tokenizer.encode_plus(
            [token], is_split_into_words=True, add_special_tokens=False
        )["input_ids"][0]

    def batch_decode(self, ids: np.ndarray, **kwargs) -> List[str]:
        return self._tokenizer.batch_decode(ids, **kwargs)

    def unwrap_tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer
