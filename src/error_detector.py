import difflib

from fuzzywuzzy import fuzz
from typing import Any

from src.audio_processor import AudioProcessor
from src.structures import Settings, FileError, ErrorType
from src.text_processor import TextProcessor

from src.utils import count_sublist_occurrences


class ErrorDetector:
    _settings: Settings
    _audio_processor: AudioProcessor
    _text_processor: TextProcessor

    errors: list[FileError] = []
    combined_silent_durations: float

    def __init__(self, whisper_result: dict[str, Any], settings: Settings):
        self._settings = settings
        self._audio_processor = AudioProcessor(settings)
        self._text_processor = TextProcessor(whisper_result, settings)

    @property
    def combined_silent_durations(self) -> float:
        return self._audio_processor.combined_silent_durations

    def _handle_replace(self, i1: int, i2: int, j1: int, j2: int) -> None:
        original_tokens = self._text_processor.original_tokens
        translated_tokens = self._text_processor.translated_tokens
        translated_words = self._text_processor.translated_words
        token_pairs = zip(
            self._text_processor.prepared_translated_tokens[i1:i2],
            self._text_processor.prepared_original_tokens[j1:j2]
        )

        i_concat = "".join(translated_tokens[i1:i2])
        j_concat = "".join(original_tokens[j1:j2])
        if (
            self._settings.token_similarity_ratio_threshold
            <= fuzz.ratio(i_concat, j_concat)
        ):
            return

        for index, (trans, orig) in enumerate(token_pairs):
            word = translated_words[i1 + index]
            start = word["start"]
            end = word["end"]
            confidence = word["confidence"]
            overlapping = self._audio_processor.check_sound_overlapping(start)
            high_confidence = confidence > self._text_processor.confidence_gap
            leven = fuzz.ratio(orig, trans)
            errors = {
                ErrorType.OVERLAPPING: FileError(
                    error_type=ErrorType.OVERLAPPING,
                    interval=(start, end),
                    correction=original_tokens[j1 + index]
                ),
                ErrorType.FACTUAL: FileError(
                    error_type=ErrorType.FACTUAL,
                    interval=(start, end),
                    correction=original_tokens[j1 + index]
                ),
                ErrorType.DICTION: FileError(
                    error_type=ErrorType.DICTION,
                    interval=(start, end),
                    correction=original_tokens[j1 + index]
                )
            }
            if leven >= self._settings.token_similarity_ratio_threshold:
                if not high_confidence and overlapping:
                    self.errors.append(errors[ErrorType.OVERLAPPING])
            else:
                if overlapping:
                    self.errors.append(errors[ErrorType.OVERLAPPING])
                else:
                    if high_confidence:
                        self.errors.append(errors[ErrorType.FACTUAL])
                    else:
                        self.errors.append(errors[ErrorType.DICTION])

        # After handling the part where ith token maps into jth token,
        # we need to analyze the tails of sublists
        i_len = i2 - i1
        j_len = j2 - j1
        i_j_diff = i_len - j_len
        # Skip the == 1 case as it often yields false positive results
        if i_j_diff > 1:
            if count_sublist_occurrences(
                translated_tokens, translated_tokens[i1 + i_j_diff:i2]
            ) > 1:
                self.errors.append(
                    FileError(
                        error_type=ErrorType.DUPLICATE,
                        interval=(
                            translated_words[i1 + i_j_diff]["start"],
                            translated_words[i2 - 1]["end"]
                        )
                    )
                )
            else:
                self.errors[-1]._interval = (
                    self.errors[-1]._interval[0],
                    translated_words[i2 - 1]["end"]
                )
        if i_j_diff < 0:
            ts = translated_words[i2]["end"]
            self.errors.append(
                FileError(
                    error_type=ErrorType.MISSING,
                    interval=(ts, ts),
                    correction=" ".join(original_tokens[j2 + i_j_diff:j2])
                )
            )

    def _handle_delete(self, i1: int, i2: int, j1: int, j2: int) -> None:
        translated_tokens = self._text_processor.translated_tokens
        translated_words = self._text_processor.translated_words
        if (
            count_sublist_occurrences(
                translated_tokens, translated_tokens[i1:i2]
            )
        ) > 1:
            self.errors.append(
                FileError(
                    error_type=ErrorType.DUPLICATE,
                    interval=(
                        translated_words[i1]["start"],
                        translated_words[i2 - 1]["end"]
                    )
                )
            )
        else:
            self.errors.append(
                FileError(
                    error_type=ErrorType.FACTUAL,
                    interval=(
                        translated_words[i1]["start"],
                        translated_words[i2 - 1]["end"]
                    )
                )
            )

    def _handle_insert(self, i1: int, j1: int, j2: int) -> None:
        original_tokens = self._text_processor.original_tokens
        translated_words = self._text_processor.translated_words
        word = (
            translated_words[i1]
            if i1 < len(translated_words)
            else translated_words[i1 - 1]
        )
        ts = word["end"]
        self.errors.append(
            FileError(
                error_type=ErrorType.MISSING,
                interval=(ts, ts),
                correction=" ".join(original_tokens[j1:j2])
            )
        )

    def _find_errors(self) -> None:
        differ = difflib.SequenceMatcher(
            None,
            self._text_processor.prepared_translated_tokens,
            self._text_processor.prepared_original_tokens
        )
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            match tag:
                case "replace":
                    self._handle_replace(i1, i2, j1, j2)
                case "delete":
                    self._handle_delete(i1, i2, j1, j2)
                case "insert":
                    self._handle_insert(i1, j1, j2)

    def run(self) -> None:
        self._audio_processor.process()
        self._text_processor.process()
        self._find_errors()
