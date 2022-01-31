import pytest
from typing import Union
import torch
from pytest_lazyfixture import lazy_fixture
from ..fixtures import *
from ...quantus.helpers import *


@pytest.fixture
def atts_normalise_1():
    return np.array([1.0, 2.0, 3.0, 4.0, 4.0, 5.0, -1.0])


@pytest.fixture
def atts_normalise_2():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def atts_denormalise():
    return np.zeros((3, 2, 2))


@pytest.fixture
def atts_denormalise_torch():
    return torch.tensor(np.zeros((3, 2, 2)))


@pytest.mark.normalise_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("atts_normalise_1"),
            {},
            np.array([0.2, 0.4, 0.6, 0.8, 0.8, 1.0, -0.2]),
        )
    ],
)
def test_normalise_by_max(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = normalise_by_max(a=data)
    assert all(o == e for o, e in zip(out, expected)), "Test failed."


@pytest.mark.normalise_func
@pytest.mark.parametrize(
    "data,params,expected",
    [(lazy_fixture("atts_normalise_2"), {}, np.array([0.2, 0.4, 0.6, 0.8, 1.0]))],
)
def test_normalise_if_negative(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = normalise_by_negative(a=data)
    assert all(o == e for o, e in zip(out, expected)), "Test failed."


@pytest.mark.normalise_func
@pytest.mark.parametrize(
    "data,params,expected",
    [
        (
            lazy_fixture("atts_denormalise"),
            {},
            np.array(
                [
                    [[0.485, 0.485], [0.485, 0.485]],
                    [[0.456, 0.456], [0.456, 0.456]],
                    [[0.406, 0.406], [0.406, 0.406]],
                ]
            ),
        ),
        (
            [1, 2],
            {},
            [1, 2],
        ),
        (
            lazy_fixture("atts_denormalise_torch"),
            {"nr_channels": 3, "img_size": 2},
            torch.tensor(
                np.array(
                    [
                        [[0.485, 0.485], [0.485, 0.485]],
                        [[0.456, 0.456], [0.456, 0.456]],
                        [[0.406, 0.406], [0.406, 0.406]],
                    ]
                )
            ),
        ),
    ],
)
def test_denormalise(
    data: np.ndarray, params: dict, expected: Union[float, dict, bool]
):
    out = denormalise(img=data, **params)
    if isinstance(data, list):
        assert out == expected, "Test failed."
        return
    assert all(
        o == e for o, e in zip(out.flatten(), expected.flatten())
    ), "Test failed."
