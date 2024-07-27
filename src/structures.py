from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Optional


class ErrorType(StrEnum):
    MISSING = auto()
    DUPLICATE = auto()
    OVERLAPPING = auto()
    FACTUAL = auto()
    DICTION = auto()


class ModelType(StrEnum):
    MEDIUM = auto()
    LARGE = auto()


class SampleType(StrEnum):
    CLEAR = auto()
    OVERLAPPING = auto()
    LONG_OVERLAPPING = auto()
    DIFFERENT_ERRORS = auto()


class FileError:
    _error_type: ErrorType
    _interval: tuple[float, float]
    _correction: Optional[str]

    def __init__(
        self, error_type: ErrorType,
        interval: tuple[float, float],
        correction: Optional[str] = None,
    ):
        self._error_type = error_type
        self._interval = interval
        self._correction = correction

    def __str__(self):
        s = f"{self._error_type.name}: {self._interval}"
        if self._correction:
            s = f"{s} -> {self._correction}"
        return s


@dataclass
class Settings:
    silence_threshold: float
    overlapping_threshold: float
    confidence_threshold: float
    token_similarity_ratio_threshold: int
    sample_number: int
    sample_type: SampleType
