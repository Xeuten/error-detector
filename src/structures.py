from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Optional


class ErrorType(StrEnum):
    """Enum class to represent the type of the error."""

    MISSING = auto()
    DUPLICATE = auto()
    OVERLAPPING = auto()
    FACTUAL = auto()
    DICTION = auto()


class SampleType(StrEnum):
    """Enum class to represent the type of the sample."""

    CLEAR = auto()
    OVERLAPPING = auto()
    LONG_OVERLAPPING = auto()
    DIFFERENT_ERRORS = auto()


class FileError:
    """Class to represent an error in a file."""

    _error_type: ErrorType
    _interval: tuple[float, float]
    _correction: Optional[str]

    def __init__(
        self,
        error_type: ErrorType,
        interval: tuple[float, float],
        correction: Optional[str] = None,
    ):
        """Initializes the FileError with the given error type, interval, and correction."""
        self._error_type = error_type
        self._interval = interval
        self._correction = correction

    def __str__(self):
        """Returns the string representation of the FileError."""
        s = f"{self._error_type.name}: {self._interval}"
        if self._correction:
            s = f"{s} -> {self._correction}"
        return s


class ModelType(StrEnum):
    """Enum class to represent the type of the Whisper model."""

    MEDIUM = auto()
    LARGE = auto()


@dataclass
class Settings:
    """Data class to represent the settings."""

    silence_threshold: float
    overlapping_threshold: float
    confidence_threshold: float
    token_similarity_ratio_threshold: int
    sample_number: int
    sample_type: SampleType
    model_type: ModelType
