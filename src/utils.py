import subprocess
from typing import Any


def tokenize(text: str) -> list[str]:
    return text.split()


def count_sublist_occurrences(big_list: list[Any], small_list: list[Any]) -> int:
    count = i = 0
    small_len = len(small_list)
    big_len = len(big_list)
    while i <= big_len - small_len:
        if big_list[i : i + small_len] == small_list:
            count += 1
            i += small_len
        else:
            i += 1
    return count


def prepare_audio(audio_path: str) -> str:
    path_split = audio_path.split(".")
    normalized_audio_path = f".{path_split[1]}_normalized.{path_split[2]}"
    command = [
        "ffmpeg",
        "-i",
        audio_path,
        "-filter:a",
        "speechnorm",
        normalized_audio_path,
    ]
    subprocess.run(command)
    return normalized_audio_path
