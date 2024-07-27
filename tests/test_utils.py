import whisper_timestamped as whisper

from src.error_detector import ErrorDetector
from src.structures import Settings
from src.utils import prepare_audio


def run(settings: Settings) -> ErrorDetector:
    normalized_audio_path = prepare_audio(
        f"./src/samples/sample_{settings.sample_number}_{settings.sample_type}.wav"
    )
    model = whisper.load_model("large", device="cuda")
    audio = whisper.load_audio(normalized_audio_path)
    result = whisper.transcribe(
        model, audio, language="ru", remove_punctuation_from_words=True
    )
    detector = ErrorDetector(result, settings)
    detector.run()
    return detector
