import pytest

from main import run
from src.settings import detector_settings
from src.structures import ErrorType


@pytest.mark.parametrize(
    "sample_number, sample_type, expected_counts",
    [
        pytest.param(
            1,
            "long_overlapping",
            {"overlapping": 9},
            marks=pytest.mark.l_overlapping_1,
        ),
        pytest.param(
            2,
            "overlapping",
            {"overlapping": 15},
            marks=pytest.mark.l_overlapping_2,
        ),
        pytest.param(
            3,
            "overlapping",
            {"overlapping": 2},
            marks=pytest.mark.l_overlapping_3,
        ),
        pytest.param(
            4,
            "overlapping",
            {"overlapping": 3},
            marks=pytest.mark.l_overlapping_4,
        ),
        pytest.param(
            5,
            "overlapping",
            {"overlapping": 0},
            marks=pytest.mark.l_overlapping_5,
        ),
    ]
)
def test_detector_overlapping(
    sample_number: int, sample_type: str, expected_counts: dict[str, int]
) -> None:
    settings = detector_settings.copy()
    settings.sample_number = sample_number
    settings.sample_type = sample_type

    detector = run(settings)
    error_counts = {error._error_type: 0 for error in detector.errors}
    for error in detector.errors:
        error_counts[error._error_type] += 1

    assert error_counts[ErrorType.OVERLAPPING] == expected_counts["overlapping"]
