from enum import Enum

class ShuffleLabelStrategy(Enum):
    CONSTRAINED_ERRORS_ONLY = 0
    ERRORS_ONLY = 1
    ALL_TOKENS = 2
