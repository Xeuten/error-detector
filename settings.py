from decouple import config

from src.structures import Settings

detector_settings = Settings(
    sample_number=config("SAMPLE_NUMBER", cast=int, default=1),
    sample_type=config("SAMPLE_TYPE", default="clear"),
    silence_threshold=config("SILENCE_THRESHOLD", cast=float),
    overlapping_threshold=config("OVERLAPPING_THRESHOLD", cast=float),
    confidence_threshold=config("CONFIDENCE_THRESHOLD", cast=float),
    token_similarity_ratio_threshold=config(
        "TOKEN_SIMILARITY_RATIO_THRESHOLD", cast=int
    ),
    model_type=config("MODEL_TYPE"),
)
