# import enum
from pathlib import Path

EMOTIONS = ["Anger", "Boredom", "Calmness", "Concentration",
            "Determination", "Excitement", "Interest", "Sadness", "Tiredness"]

TYPES = ["Anger", "Boredom", "Calmness", "Concentration",
         "Determination", "Excitement", "Interest", "Sadness", "Tiredness"]

# mapping of string types to integer class indices
MAP_TYPES = {t: i for i, t in enumerate(TYPES)}
INVERSE_MAP_VOCAL_TYPES = {i: t for i, t in enumerate(TYPES)}
