import numpy as np


def empty_mask(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)
