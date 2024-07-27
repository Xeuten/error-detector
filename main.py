import whisper_timestamped as whisper

from src.error_detector import ErrorDetector
from src.settings import detector_settings
from src.utils import prepare_audio

normalized_audio_path = prepare_audio(
    f"./src/samples/sample_{detector_settings.sample_number}_{detector_settings.sample_type}.wav"
)
model = whisper.load_model("large")
audio = whisper.load_audio(normalized_audio_path)
result = whisper.transcribe(
    model, audio, language="ru", remove_punctuation_from_words=True
)
detector = ErrorDetector(result, detector_settings)
detector.run()
print("Errors:", *detector.errors, sep="\n")
print(f"Combined silent durations: {detector.combined_silent_durations:.2f} seconds")
