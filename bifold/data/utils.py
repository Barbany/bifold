import ast

import numpy as np

DENG_CAMERA_PARAMS = {
    "default_camera": {
        "pos": np.array([0.0, 0.65, 0.0]),
        "angle": np.array([0.0, -1.57079633, 0.0]),
        "width": 720,
        "height": 720,
    }
}


def get_mask_from_depth(depth):
    # generate a mask
    mask = depth.copy()
    mask[mask > 0.996] = 0
    mask[mask != 0] = 1
    return mask


def parse_list_string(s):
    try:
        # Using ast.literal_eval to safely evaluate the string as a Python literal
        return ast.literal_eval(s)
    except (SyntaxError, ValueError):
        # If there's a syntax error or value error, return None
        return None
