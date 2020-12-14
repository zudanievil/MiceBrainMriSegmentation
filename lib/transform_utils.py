"""
utility functions for ndarray transformations
"""

import numpy as np
import skimage.transform

_LOC = dict()
_GLOB = dict()


def transform_factory(rotate_w = 0, center=(0, 0), pan = (0, 0),
                     scale = (0, 0), shear = (0, 0), shift = (0, 0)) -> skimage.transform.FundamentalMatrixTransform: #TODO: add center_w; fix pan
    """
    note that all parameters are in uv-coordincate style (top left corner is 0, 0, rotation is clockwise)
    """
    rotate = (np.sin(rotate_w), np.cos(rotate_w))
    h = [
        # v-row
        rotate[1],
        rotate[0] + shear[1],
        (scale[1] - rotate[0]) * center[1] - rotate[0] * center[0] + shift[1],
        # u-row
        -rotate[0],
        rotate[1] + shear[0],
        (scale[0] - rotate[1]) * center[0] + rotate[0] * center[1] + shift[0],
        # w-row
        pan[1],
        pan[0],
        1,
    ]
    h = np.array(h, dtype=np.float).reshape((3, 3))
    h = skimage.transform.FundamentalMatrixTransform(matrix=h)
    return h


def bbox_crop(image: np.ndarray, bbox) -> np.ndarray:
    return image[bbox[0]:bbox[1], bbox[2]:bbox[3]]


def resize(image: np.ndarray, shape: (int, int)) -> np.ndarray:
    return skimage.transform.resize(image, shape, **_LOC['resize_kw'])


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    return skimage.transform.rotate(image, angle, **_LOC['rotate_kw'])

