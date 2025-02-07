from enum import Enum


class DataType(Enum):
    """
    This enum represents the different types of data that a metric implementation currently supports.

        - IMAGE: Represents image data.
        - TABULAR: Represents tabular data.
        - TEXT: Represents text data.
    """

    IMAGE = "image"  # 3D data
    TIMESERIES = "time-series"  # 1D data
    TABULAR = "tabular"  # 2D data
    TEXT = "text"


class ModelType(Enum):
    """
    This enum represents the different types of models that a metric can work with.

        - TORCH: Represents PyTorch models.
        - TF: Represents TensorFlow models.
    """

    TORCH = "torch"
    TF = "tensorflow"


class ScoreDirection(Enum):
    """
    This enum represents the direction that the score of a metric should go in for better results.

        - HIGHER: Higher scores are better.
        - LOWER: Lower scores are better.
    """

    HIGHER = "higher"
    LOWER = "lower"


class EvaluationCategory(Enum):
    """
    This enum represents different categories of explanation quality for XAI algorithms.

        - FAITHFULNESS: Indicates how well the explanation reflects the true features used by the model.
        - ROBUSTNESS: Represents the degree to which the explanation remains consistent under small perturbations in the input.
        - RANDOMISATION: Measures the quality of the explanation in terms of difference in explanation when randomness is introduced.
        - COMPLEXITY: Refers to how easy it is to understand the explanation. Lower complexity is usually better.
        - LOCALISATION: Refers to how consistently the explanation points out the parts of the input as defined in a ground-truth segmentation mask.
        - AXIOMATIC: Represents the quality of the explanation in terms of well-defined axioms.
    """

    FAITHFULNESS = "Faithfulness"
    ROBUSTNESS = "Robustness"
    RANDOMISATION = "Randomisation"
    COMPLEXITY = "Complexity"
    LOCALISATION = "Localisation"
    AXIOMATIC = "Axiomatic"
    UNIFIED = "Unified"
    NONE = "None"
