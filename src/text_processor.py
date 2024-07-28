import re
from functools import reduce
from typing import Any, Optional

import numpy as np

from src.samples.samples import samples
from src.structures import Settings
from src.utils import tokenize


class TextProcessor:
    """This class determines the confidence gap, translated words, and tokens."""

    _original_text: str
    _translated_text: str
    _segments: list[dict[str, Any]]
    _confidence_threshold: float

    confidence_gap: float
    translated_words: list[dict[str, Any]]
    original_tokens: list[str]
    translated_tokens: list[str]
    prepared_original_tokens: list[str]
    prepared_translated_tokens: list[str]

    def __init__(
        self,
        whisper_result: dict[str, Any],
        settings: Settings,
        text_path: Optional[str] = None,
    ):
        """Initializes the TextProcessor with the given whisper result and settings."""
        self._original_text = samples[f"sample_{settings.sample_number}"]
        if text_path:
            with open(text_path, "r", encoding="utf-8") as file:
                self._original_text = file.read()
        self._translated_text = whisper_result["text"]
        self._segments = whisper_result["segments"]
        self._confidence_threshold = settings.confidence_threshold

    def _prepare_text(self, text: str) -> str:
        """Prepares the given text by removing punctuation and normalizing spaces."""
        text = re.sub(r"[^\w\d\s]", "", text, flags=re.UNICODE)
        return " ".join(tokenize(text))

    def _prepare_word(self, word: str):
        """Prepares the given word by replacing 'ё' with 'е' and converting it to lowercase."""
        return word.replace("ё", "е").lower()

    def _set_translated_words(self) -> None:
        """Sets the translated words by combining the words from all segments."""
        words_lists = [segment["words"] for segment in self._segments]
        self.translated_words = reduce(lambda acc, words: acc + words, words_lists, [])

    def _set_confidence_gap(self) -> None:
        """Sets the confidence gap by calculating the mean confidence and standard deviation."""
        confidences = [
            word["confidence"]
            for word in self.translated_words[1 : len(self.translated_words) - 1]
        ]

        # I distinguish between different confidence values in the same way as I did
        # with non-silent intervals and overlapping intervals
        mean_confidence = np.mean(confidences)
        std_dev_confidence = np.std(confidences)
        self.confidence_gap = (
            mean_confidence - self._confidence_threshold * std_dev_confidence
        )

    def _set_tokens(self) -> None:
        """Sets the original and translated tokens and their prepared versions."""
        translated = self._prepare_text(self._translated_text)
        original = self._prepare_text(self._original_text)
        self.original_tokens = tokenize(original)
        self.translated_tokens = tokenize(translated)
        self.prepared_original_tokens = [
            self._prepare_word(token) for token in self.original_tokens
        ]
        self.prepared_translated_tokens = [
            self._prepare_word(token) for token in self.translated_tokens
        ]

    def process(self) -> None:
        """Processes the text by setting the translated words, confidence gap, and tokens."""
        self._set_translated_words()
        self._set_confidence_gap()
        self._set_tokens()
