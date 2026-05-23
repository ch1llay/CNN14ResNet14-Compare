"""НИР-3: Сравнительное исследование CNN14 vs ResNet18 на GTZAN."""

__version__ = "0.1.0"

GENRES = (
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
)
NUM_CLASSES = len(GENRES)
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRES)}
