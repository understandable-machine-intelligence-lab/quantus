import numpy as np
import pytest

import quantus.nlp as qn


@pytest.mark.order("last")
@pytest.mark.nlp
@pytest.mark.evaluate_func
def test_tf_model(tf_sst2_model, sst2_dataset):
    metrics = {
        "Avg-Sesnitivity": qn.AvgSensitivity(nr_samples=5),
        "Max-Sensitivity": qn.MaxSensitivity(nr_samples=5),
        "RIS": qn.RelativeInputStability(nr_samples=5),
        "RandomLogit": qn.RandomLogit(num_classes=2),
    }

    call_kwargs = {
        "explain_func_kwargs": {"method": "GradXInput"},
        "batch_size": 8,
        "Max-Sensitivity": {
            "explain_func_kwargs": {"method": "SHAP", "call_kwargs": {"max_evals": 5}}
        },
        "RIS": [
            {"explain_func_kwargs": {"method": "IntGrad", "num_steps": 5}},
            {"explain_func_kwargs": {"method": "IntGrad", "num_steps": 7}},
        ],
    }
    # Just check that it doesn't fail with expected inputs.
    result = qn.evaluate(metrics, tf_sst2_model, sst2_dataset, call_kwargs=call_kwargs)
    result_ris = result["RIS"]
    # CHeck list of args returns list of scores
    assert isinstance(result_ris, list)
    assert len(result_ris) == 2
    assert isinstance(result_ris[0], np.ndarray)
    assert isinstance(result_ris[1], np.ndarray)
