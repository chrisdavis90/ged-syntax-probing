from enum import Enum

class GEDLabel(Enum):
    BINARY = 0
    OP = 1
    MAIN = 2
    ALL = 3

def num_labels_to_ged_label(num_labels: int):
    if num_labels in [2, 3]:
        return GEDLabel.BINARY
    elif num_labels in [4, 5]:
        return GEDLabel.OP
    elif num_labels in [25, 26]:
        return GEDLabel.MAIN
    else:
        return GEDLabel.ALL

