import pytest

from main import run
from settings import detector_settings
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
            "long_overlapping",
            {"overlapping": 15},
            marks=pytest.mark.l_overlapping_2,
        ),
        pytest.param(
            3,
            "long_overlapping",
            {"overlapping": 2},
            marks=pytest.mark.l_overlapping_3,
        ),
        pytest.param(
            4,
            "long_overlapping",
            {"overlapping": 3},
            marks=pytest.mark.l_overlapping_4,
        ),
        pytest.param(
            5,
            "long_overlapping",
            {"overlapping": 0},
            marks=pytest.mark.l_overlapping_5,
        ),
    ]
)
def test_detector_overlapping(
    sample_number: int, sample_type: str, expected_counts: dict[str, int]
) -> None:
    settings = detector_settings
    settings.sample_number = sample_number
    settings.sample_type = sample_type

    detector = run(settings)
    error_counts = {error._error_type: 0 for error in detector.errors}
    for error in detector.errors:
        error_counts[error._error_type] += 1

    assert error_counts[ErrorType.OVERLAPPING] == expected_counts["overlapping"]
