from typing import List
import numpy as np
import pytest
from quantus.nlp import explain


def unknown_token_baseline_function(_) -> np.ndarray:
    return np.load("tests/assets/nlp/unknown_token_embedding.npy")


@pytest.mark.nlp
@pytest.mark.tf_model
@pytest.mark.explain_func
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "batch_interpolated_inputs": True},
        {"method": "IntGrad", "baseline_fn": unknown_token_baseline_function},
        {"method": "NoiseGrad", "explain_fn": "GradXInput", "n": 2, "m": 2},
        {"method": "NoiseGrad++", "explain_fn": "GradXInput", "n": 2, "m": 2},
        {"method": "LIME", "call_kwargs": {"num_samples": 5}},
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad iterative",
        "IntGrad batched",
        "IntGrad [UNK] baseline",
        "NoiseGrad",
        "NoiseGrad++",
        "LIME",
        "SHAP",
    ],
)
def test_tf_model(tf_sst2_model, sst2_dataset, kwargs):
    y_batch = tf_sst2_model.predict(sst2_dataset).argmax(axis=-1)
    a_batch = explain(tf_sst2_model, sst2_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)


@pytest.mark.nlp
@pytest.mark.explain_func
@pytest.mark.pytorch_model
@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "GradNorm"},
        {"method": "GradXInput"},
        {"method": "IntGrad"},
        {"method": "IntGrad", "batch_interpolated_inputs": True},
        {"method": "IntGrad", "baseline_fn": unknown_token_baseline_function},
        {
            "method": "NoiseGrad",
            "explain_fn": "GradXInput",
            "init_kwargs": {"n": 2},
        },
        {
            "method": "NoiseGrad++",
            "explain_fn": "GradXInput",
            "init_kwargs": {"n": 2, "m": 2},
        },
        pytest.param(
            {"method": "LIME", "call_kwargs": {"num_samples": 5}},
            marks=pytest.mark.xfail,
        ),
        {"method": "SHAP", "call_kwargs": {"max_evals": 5}},
    ],
    ids=[
        "GradNorm",
        "GradXInput",
        "IntGrad iterative",
        "IntGrad batched",
        "IntGrad [UNK] baseline",
        "NoiseGrad",
        "NoiseGrad++",
        "LIME",
        "SHAP",
    ],
)
def test_torch_fnet_model(torch_sst2_model, sst2_dataset, kwargs):
    y_batch = torch_sst2_model.predict(sst2_dataset).argmax(axis=-1)
    a_batch = explain(torch_sst2_model, sst2_dataset, y_batch, **kwargs)
    assert len(a_batch) == len(y_batch)
    for tokens, scores in a_batch:
        assert isinstance(tokens, List)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1
