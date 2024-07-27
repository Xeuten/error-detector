import re
from functools import reduce
from typing import Any

import numpy as np

from src.samples.samples import samples
from src.structures import Settings
from src.utils import tokenize


class TextProcessor:
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

    def __init__(self, whisper_result: dict[str, Any], settings: Settings):
        self._original_text = samples[f"sample_{settings.sample_number}"]
        self._translated_text = whisper_result["text"]
        self._segments = whisper_result["segments"]
        self._confidence_threshold = settings.confidence_threshold

    def _prepare_text(self, text: str) -> str:
        text = re.sub(r"[^\w\d\s]", "", text, flags=re.UNICODE)
        return " ".join(tokenize(text))

    def _prepare_word(self, word: str):
        return word.replace("ё", "е").lower()

    def _set_translated_words(self) -> None:
        words_lists = [segment["words"] for segment in self._segments]
        self.translated_words = reduce(lambda acc, words: acc + words, words_lists, [])

    def _set_confidence_gap(self) -> None:
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
        self._set_translated_words()
        self._set_confidence_gap()
        self._set_tokens()
