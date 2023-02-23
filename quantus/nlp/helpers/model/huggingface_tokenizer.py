from typing import List, Dict
import numpy as np

from transformers import PreTrainedTokenizerBase
from quantus.nlp.helpers.model.text_classifier import Tokenizer


class HuggingFaceTokenizer(Tokenizer):

    """A wrapper around HuggingFace's hub tokenizers, which encapsulates common functionality used in Quantus."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def tokenize(self, text: List[str]) -> Dict[str, np.ndarray]:
        return self.tokenizer(text, padding="longest", return_tensors="np").data

    def convert_ids_to_tokens(self, ids: np.ndarray) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def token_id(self, token: str) -> int:
        return self.tokenizer.encode_plus(
            [token], is_split_into_words=True, add_special_tokens=False
        )["input_ids"][0]
