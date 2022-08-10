from typing import Union

import numpy as np
import optax
import pytest
from pytest_lazyfixture import lazy_fixture

from ..fixtures import *
from ...quantus.metrics import *
from ...quantus.helpers import *
from ...quantus.helpers.explanation_func import explain
from quantus.metrics.robustness_metrics import ris_objective


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "perturb_std": 0.1,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "perturb_std": 0.1,
                    "nr_samples": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "perturb_std": 0.1,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "perturb_std": 0.1,
                    "nr_samples": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "perturb_std": 0.1,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "a_batch_generate": False,
                    "return_aggregate": True,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "perturb_std": 0.1,
                    "nr_samples": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_local_lipschitz_estimate(
        model: ModelInterface,
        data: np.ndarray,
        params: dict,
        expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = LocalLipschitzEstimate(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )
    assert scores is not None, "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "perturb_radius": 0.2,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "perturb_radius": 0.2,
                    "nr_samples": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "perturb_radius": 0.2,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "perturb_radius": 0.2,
                    "nr_samples": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "perturb_radius": 0.2,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "perturb_radius": 0.2,
                    "nr_samples": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": True,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {
                    "perturb_radius": 0.2,
                    "nr_samples": 10,
                    "return_aggregate": True,
                    "img_size": 28,
                    "nr_channels": 1,
                    "explain_func": explain,
                    "method": "Gradient",
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "abs": True,
                    "normalise": True,
                },
                {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_max_sensitivity(
        model: ModelInterface,
        data: np.ndarray,
        params: dict,
        expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = MaxSensitivity(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )

    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert np.all(
            ((s >= expected["min"]) & (s <= expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "perturb_radius": 0.2,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                    "return_aggregate": True,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "perturb_radius": 0.2,
                    "nr_samples": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "perturb_radius": 0.2,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "perturb_radius": 0.2,
                    "nr_samples": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "perturb_radius": 0.2,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "perturb_radius": 0.2,
                    "nr_samples": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "a_batch_generate": False,
                    "return_aggregate": True,
                },
                {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_avg_sensitivity(
        model: ModelInterface,
        data: np.ndarray,
        params: dict,
        expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None
    scores = AvgSensitivity(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )
    if isinstance(expected, float):
        assert all(s == expected for s in scores), "Test failed."
    else:
        assert np.all(
            ((s >= expected["min"]) & (s <= expected["max"])) for s in scores
        ), "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "nr_steps": 10,
                    "patch_size": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"exception": ValueError},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "nr_steps": 10,
                    "patch_size": 7,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "nr_steps": 10,
                    "patch_size": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                    "return_aggregate": True,
                },
                {"exception": ValueError},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "nr_steps": 10,
                    "patch_size": 7,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": False,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_1d_3ch_conv_model"),
                lazy_fixture("almost_uniform_1d_no_abatch"),
                {
                    "nr_steps": 10,
                    "patch_size": 10,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "a_batch_generate": False,
                },
                {"exception": ValueError},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "nr_steps": 10,
                    "patch_size": 7,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": True,
                    "display_progressbar": True,
                    "a_batch_generate": False,
                    "return_aggregate": True,
                },
                {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_continuity(
        model: ModelInterface,
        data: np.ndarray,
        params: dict,
        expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = Continuity(**params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **params,
            )
        return

    scores = Continuity(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )
    assert scores is not None, "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "nr_steps": 10,
                    "patch_size": 7,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "discretise_func": floating_points,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "nr_steps": 10,
                    "patch_size": 7,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "discretise_func": sign,
                    "a_batch_generate": True,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "nr_steps": 10,
                    "patch_size": 7,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "discretise_func": top_n_sign,
                    "a_batch_generate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
        (
                lazy_fixture("load_mnist_model"),
                lazy_fixture("load_mnist_images"),
                {
                    "nr_steps": 10,
                    "patch_size": 7,
                    "explain_func": explain,
                    "method": "Saliency",
                    "disable_warnings": False,
                    "display_progressbar": False,
                    "discretise_func": rank,
                    "a_batch_generate": False,
                    "return_aggregate": False,
                },
                {"min": 0.0, "max": 1.0},
        ),
    ],
)
def test_consistency(
        model: ModelInterface,
        data: np.ndarray,
        params: dict,
        expected: Union[float, dict, bool],
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    if params.get("a_batch_generate", True):
        explain = params["explain_func"]
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **params,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = Consistency(**params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **params,
            )
        return

    scores = Consistency(**params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **params,
    )[0]
    assert (scores >= expected["min"]) & (scores <= expected["max"]), "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params",
    [
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "Gradient"}
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "IntegratedGradients"}
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "InputXGradient"}
        ),
        (
            lazy_fixture("load_mnist_model_tf"),
            lazy_fixture("load_mnist_images_tf"),
            {"method": "Occlusion"}
        )
    ]
)
def test_ris_objective(
        model: ModelInterface,
        data: np.ndarray,
        params
):
    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )

    xs_batch = perturb_func.gaussian_noise(x_batch, list(range(124)), [0])
    logits = model.predict(xs_batch)
    ys_batch = tf.argmax(tf.math.softmax(logits, axis=0), axis=1).numpy()

    ex = explain(model, x_batch, y_batch, **params)
    exs = explain(model, xs_batch, ys_batch, **params)

    result = ris_objective(x_batch, xs_batch, ex, exs, 1e-5)
    result = result.to_py()
    print(f'{result = }')

    assert (result != np.nan).all(), "Test failed."


@pytest.mark.robustness
@pytest.mark.parametrize(
    "model,data,params,expected",
    [
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"method": "Gradient"},
                # The results from original paper were in these bounds
                {"min": 0, "max": 15}
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"method": "IntegratedGradients"},
                # The results from original paper were in these bounds
                {"min": 0, "max": 15}
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"method": "InputXGradient"},
                # The results from original paper were in these bounds
                {"min": 0, "max": 15}
        ),
        (
                lazy_fixture("load_mnist_model_tf"),
                lazy_fixture("load_mnist_images_tf"),
                {"method": "Occlusion"},
                # The results from original paper were in these bounds
                {"min": 0, "max": 15}
        ),

    ]
)
def test_relative_input_stability(
        model: ModelInterface,
        data: np.ndarray,
        params,
        expected
):

    x_batch, y_batch = (
        data["x_batch"],
        data["y_batch"],
    )
    ris = RelativeInputStability()

    result = ris(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        perturb_func=perturb_func.random_noise,
        explain_func=explain,
        **params
    )

    print(f'{result = }')
    assert (result >= expected["min"]).all(), "Test failed."
    assert (result <= expected["max"]).all(), "Test failed."




def test_relative_output_stability():
    pass


def test_relative_representation_stability():
    pass
