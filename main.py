import sys

import whisper_timestamped as whisper

from settings import detector_settings
from src.error_detector import ErrorDetector
from src.structures import Settings
from src.utils import prepare_audio


def run(settings: Settings = detector_settings) -> ErrorDetector:
    audio_path = (
        f"./src/samples/sample_{settings.sample_number}_{settings.sample_type}.wav"
    )
    text_path = None
    if len(sys.argv) == 3:
        audio_path = sys.argv[1]
        text_path = sys.argv[2]

    normalized_audio_path = prepare_audio(audio_path)
    model = whisper.load_model(settings.model_type, device="cuda")
    audio = whisper.load_audio(normalized_audio_path)
    result = whisper.transcribe(
        model, audio, language="ru", remove_punctuation_from_words=True
    )
    detector = ErrorDetector(result, settings, audio_path, text_path)
    detector.run()
    return detector


detector = run()
print("Errors:", *detector.errors, sep="\n")
print(f"Combined silent durations: {detector.combined_silent_durations:.2f} seconds")
