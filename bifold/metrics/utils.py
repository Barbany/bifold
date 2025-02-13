import numpy as np


def iou(mask1, mask2):
    intersection = np.count_nonzero(mask1 * mask2)
    union = np.count_nonzero(mask1 + mask2)
    return intersection / union * 100
