import numpy as np
import skimage.segmentation as sks
import skimage.transform as skt
import skimage.filters as skf
from ..core import info_classes
from ..utils import linalg_utils


def main(
    image_info: info_classes.ImageInfo, ontology_info: info_classes.OntologyInfo
) -> ("np.ndarray[bool]", "np.ndarray[bool]"):
    """returns 2 boolean masks: first for animal brain, second for the whole head"""
    image = image_info.image()
    image = linalg_utils.sigmoid_with_quantiles(image, quantiles=(0.2, 0.8))
    brain_coo = get_brain_coordinates(image_info.metadata(), image.shape)
    brain_mask = brain_mask_to_image(
        open_root_mask(ontology_info), brain_coo, image.shape
    )
    head_mask = make_head_mask(image, brain_mask)
    return brain_mask, head_mask


def get_brain_coordinates(
    meta: dict, image_shape: tuple
) -> "np.ndarray[float]":
    """uses metadata to determine bounding box for the brain,
    returns it as (x, y) coordinate pairs (an array of shape (n, 2))"""
    l = meta["lbbox"]
    r = meta["rbbox"]
    brain_bbox = (l[0] + r[0]) / 2, (l[1] + r[1]) / 2, l[2], r[3]
    brain_coo = linalg_utils.bbox_to_coords(*brain_bbox)
    angle = -meta["rotation"]
    center = (
        image_shape[1] / 2 - 0.5,
        image_shape[0] / 2 - 0.5,
    )  # horizontal, vertical
    brain_coo = linalg_utils.rotate_coords(
        brain_coo, center, angle, degrees=True
    )
    return brain_coo


def open_root_mask(oi: info_classes.OntologyInfo) -> "np.ndarray[bool]":
    """opens and returns mask named `Root`"""
    return oi.open_mask(oi.mask_path_absolute(mask_name="Root"))


def brain_mask_to_image(
    mask: "np.ndarray[bool]", brain_coo: "np.ndarray[float]", image_shape: tuple
) -> "np.ndarray[bool]":
    """
    applies homographic transformation to map brain mask onto the actual image
    """
    mask |= np.flip(mask, axis=1)
    src_coo = linalg_utils.bbox_to_coords(0, mask.shape[0], 0, mask.shape[1])
    pad = (
        (0, image_shape[0] - mask.shape[0]),
        (0, image_shape[1] - mask.shape[1]),
    )
    mask = np.pad(mask, pad, constant_values=0)
    tform = skt.estimate_transform("affine", brain_coo, src_coo)
    mask = skt.warp(mask.astype(np.float), tform, order=3, preserve_range=True)
    return mask > 0.5


def make_head_mask(
    image: "np.ndarray[float]", brain_mask: "np.ndarray[bool]"
) -> "np.ndarray[bool]":
    """
    applies heuristic (thresholding and flooding) to obtain a mask of an animal head
    """
    x = skf.gaussian(image, sigma=2, truncate=2)
    x = np.where(x < 0.5, 1, 0)
    c = center_of_binary_image(brain_mask)
    mask = sks.flood(x, c)
    return mask & ~brain_mask


def center_of_binary_image(bimg: "np.ndarray[bool]") -> (int, int):
    """finds a center of a bounding box of the `True` values
    :returns UV coordinate pair"""
    rows = np.any(bimg, axis=1)
    cols = np.any(bimg, axis=0)
    u = np.where(rows)[0][[0, -1]].sum() // 2
    v = np.where(cols)[0][[0, -1]].sum() // 2
    return u, v
