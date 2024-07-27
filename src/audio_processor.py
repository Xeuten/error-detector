import librosa
import numpy as np
from numpy import ndarray

from src.structures import Settings

type Intervals = list[list[int]]
type SecondsIntervals = list[list[float]]


class AudioProcessor:
    _rms: ndarray
    _silence_threshold: float
    _overlapping_threshold: float
    _sampling_rate: int
    _overlapping_intervals: SecondsIntervals

    combined_silent_durations: float

    def __init__(self, settings: Settings):
        y, self._sampling_rate = librosa.load(
            f"/src/samples/sample_{settings.sample_number}_{settings.sample_type}.wav",
            sr=None,
        )
        self._rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        self._silence_threshold = settings.silence_threshold
        self._overlapping_threshold = settings.overlapping_threshold

    def _to_seconds(self, value: int) -> float:
        return value * 512 / self._sampling_rate

    def _to_durations(self, intervals: Intervals) -> list[float]:
        return [self._to_seconds(end - start) for start, end in intervals]

    def _determine_intervals(self) -> tuple[Intervals, Intervals]:
        non_silent_intervals = []
        silent_intervals = []
        current_interval = []
        is_silent = False

        for i, rms_value in enumerate(self._rms):
            if rms_value > self._silence_threshold:
                if is_silent:
                    if current_interval:
                        current_interval.append(i)
                        silent_intervals.append(current_interval)
                        current_interval = []
                    is_silent = False
                if not current_interval:
                    current_interval = [i]
            else:
                if not is_silent:
                    if current_interval:
                        current_interval.append(i)
                        non_silent_intervals.append(current_interval)
                        current_interval = []
                    is_silent = True
                if not current_interval:
                    current_interval = [i]

        # handle the last interval
        if current_interval:
            current_interval.append(len(self._rms))
            if is_silent:
                silent_intervals.append(current_interval)
            else:
                non_silent_intervals.append(current_interval)

        return non_silent_intervals, silent_intervals

    def _find_overlapping_intervals(
        self,
        non_silent_intervals: Intervals,
        non_silent_durations: list[float],
    ) -> SecondsIntervals:
        mean_non_silent_duration = np.mean(non_silent_durations)
        std_dev_non_silent_duration = np.std(non_silent_durations)
        sound_overlapping = [
            [self._to_seconds(interval[0]), self._to_seconds(interval[1])]
            for interval, duration in zip(non_silent_intervals, non_silent_durations)
            if duration
            > (
                # I assume that non-silent intervals that exceed a certain amount of
                # stds are intervals with sound overlapping
                self._overlapping_threshold * std_dev_non_silent_duration
                + mean_non_silent_duration
            )
        ]
        return sound_overlapping

    def process(self) -> None:
        non_silent_intervals, silent_intervals = self._determine_intervals()
        non_silent_durations = self._to_durations(non_silent_intervals)
        self._overlapping_intervals = self._find_overlapping_intervals(
            non_silent_intervals, non_silent_durations
        )
        self.combined_silent_durations = np.sum(self._to_durations(silent_intervals))

    def check_sound_overlapping(self, interval_start: float) -> bool:
        return any(
            interval[0] <= interval_start < interval[1]
            for interval in self._overlapping_intervals
        )
