import pytest
from typing import Union
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *


@pytest.fixture
def input_pert_1d():
    return np.random.uniform(0, 0.1, size=(3, 224, 224))


@pytest.fixture
def input_zeros():
    return np.zeros(shape=(3, 224, 224))


@pytest.fixture
def input_ones_mnist():
    return np.ones(shape=(1, 28, 28))


@pytest.fixture
def input_pert_3d():
    return np.random.uniform(0, 0.1, size=(3, 224, 224))


@pytest.fixture
def input_pert_mnist():
    return np.random.uniform(0, 0.1, size=(1, 28, 28))


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected", [(lazy_fixture("input_pert_1d"), {}, True)]
)
def test_gaussian_noise(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = gaussian_noise(img=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_zeros"),
            {
                "indices": np.unravel_index([0, 2, 224, 226, 448, 450], shape=(224, 224)),
                "perturb_baseline": 1.0,
                "nr_channels": 3,
            },
            1,
        ),
        (
            lazy_fixture("input_ones_mnist"),
            {"indices": np.unravel_index([0, 2, 224, 226, 448, 450], shape=(224, 224)), "perturb_baseline": -1.0, "nr_channels": 1},
            -1,
        ),
    ],
)
def test_baseline_replacement_by_indices(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = baseline_replacement_by_indices(img=data, **params)

    if isinstance(expected, (int, float)):
        assert np.all([i == expected for i in out[((slice(0, params["nr_channels"])),) + params["indices"]]]), "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_zeros"),
            {
                "indices": np.unravel_index([0, 2, 224, 226, 448, 450], shape=(224, 224)),
                "input_shift": 1.0,
                "nr_channels": 3,
            },
            0,
        ),
        (
            lazy_fixture("input_ones_mnist"),
            {"indices": np.unravel_index([0, 2, 224, 226, 448, 450], shape=(224, 224)), "input_shift": -1.0, "nr_channels": 1},
            -1,
        ),
    ],
)
def test_baseline_replacement_by_shift(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = baseline_replacement_by_shift(img=data, **params)

    if isinstance(expected, (int, float)):
        assert np.all([i == expected for i in out[((slice(0, params["nr_channels"])),) + params["indices"]]]), "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [(lazy_fixture("input_pert_1d"), {"perturb_radius": 0.02}, True)],
)
def test_uniform_sampling(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = uniform_sampling(img=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [(lazy_fixture("input_pert_3d"), {"perturb_angle": 30, "img_size": 224}, True)],
)
def test_rotation(data: dict, params: dict, expected: Union[float, dict, bool]):
    out = rotation(img=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_pert_3d"),
            {"perturb_dx": 20, "perturb_baseline": "black", "img_size": 224},
            True,
        )
    ],
)
def test_translation_x_direction(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = translation_x_direction(img=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_pert_3d"),
            {"perturb_dx": 20, "perturb_baseline": "black", "img_size": 224},
            True,
        )
    ],
)
def test_translation_y_direction(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = translation_y_direction(img=data, **params)
    assert np.any(out != data) == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected", [(lazy_fixture("input_pert_3d"), {"perturb_dx": 20}, True)]
)
def test_no_perturbation(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = no_perturbation(img=data, **params)
    print(out == data)
    assert (out == data).all() == expected, "Test failed."


@pytest.mark.perturb_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("input_pert_3d"),
            {
                "nr_channels": 3,
                "img_size": 224,
                "blur_patch_size": 15,
                "indices": (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
            },
            {"shape": True, "values": False},
        ),
        (
            lazy_fixture("input_pert_3d"),
            {
                "nr_channels": 3,
                "img_size": 224,
                "blur_patch_size": 7,
                "indices": (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
            },
            {"shape": True, "values": False},
        ),
        (
            lazy_fixture("input_pert_mnist"),
            {
                "nr_channels": 1,
                "img_size": 28,
                "blur_patch_size": 15,
                "indices": (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
            },
            {"shape": True, "values": False},
        ),
    ],
)
def test_baseline_replacement_by_blur(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = baseline_replacement_by_blur(img=data, **params)
    assert (out.shape == data.shape) == expected["shape"], "Test failed."
    assert (out == data).all() == expected["values"], "Test failed."
