from typing import List

from typing import Callable, Optional
import numpy as np
from nlpaug.augmenter.word import SynonymAug, SpellingAug
from nlpaug.augmenter.char import KeyboardAug
from quantus.nlp.helpers.utils import value_or_default

_ApplyFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def spelling_replacement(x_batch: List[str], k: int = 3, **kwargs) -> List[str]:
    """
    Replace k words in each entry of text by alternative spelling.

    Examples
    --------

    >>> x = ["uneasy mishmash of styles and genres."]
    >>> spelling_replacement(x)
    ... ['uneasy mishmash of stiles and genres.']

    """
    aug = SpellingAug(aug_max=k, aug_min=k, **kwargs)
    return aug.augment(x_batch)


def synonym_replacement(x_batch: List[str], k: int = 3, **kwargs) -> List[str]:
    """
    Replace k words in each entry of text by synonym.

    Examples
    --------

    >>> x = ["uneasy mishmash of styles and genres."]
    >>> synonym_replacement(x)
    ... ['nervous mishmash of styles and genres.']
    """
    aug = SynonymAug(aug_max=k, aug_min=k, **kwargs)
    return aug.augment(x_batch)


def typo_replacement(x_batch: List[str], k: int = 1, **kwargs) -> List[str]:
    """
    Replace k characters in k words in each entry of text mimicking typo.

    Examples
    --------
    >>> x = ["uneasy mishmash of styles and genres."]
    >>> typo_replacement(x)
    ... ['uneasy mishmash of xtyles and genres.']
    """
    aug = KeyboardAug(
        aug_char_max=k, aug_char_min=k, aug_word_min=k, aug_word_max=k, **kwargs
    )
    return aug.augment(x_batch)


def uniform_noise(
    x_batch: np.ndarray, seed: int = 42, apply_fn: Optional[_ApplyFn] = None, **kwargs
) -> np.ndarray:
    """Apply uniform noise to arr."""
    apply_fn = value_or_default(apply_fn, lambda: lambda x, y: x + y)
    noise = np.random.default_rng(seed).uniform(size=x_batch.shape, **kwargs)
    return apply_fn(x_batch, noise)


def gaussian_noise(
    x_batch: np.ndarray, seed: int = 42, apply_fn: Optional[_ApplyFn] = None, **kwargs
) -> np.ndarray:
    """Apply gaussian noise to arr."""
    apply_fn = value_or_default(apply_fn, lambda: lambda x, y: x + y)
    noise = np.random.default_rng(seed).normal(size=x_batch.shape, **kwargs)
    return apply_fn(x_batch, noise)
