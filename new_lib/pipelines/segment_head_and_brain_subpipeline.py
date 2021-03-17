import numpy as np
import skimage.segmentation as sks
import skimage.transform as skt
import skimage.filters as skf
from ..core import info_classes
from ..utils import linalg_utils


def main(image_info: info_classes.ImageInfo, ontology_info: info_classes.OntologyInfo) \
        -> ('np.ndarray[bool]', 'np.ndarray[bool]'):
    image = image_info.image()
    image = linalg_utils.sigmoid_with_quantiles(image, quantiles=(0.2, 0.8))
    brain_coo = get_brain_coordinates(image_info, image.shape)
    brain_mask = make_brain_mask(ontology_info, brain_coo, image.shape)
    head_mask = make_head_mask(image, brain_mask)
    return brain_mask, head_mask


def get_brain_coordinates(image_info, image_shape):
    meta = image_info.metadata()
    l = meta['lbbox']
    r = meta['rbbox']
    brain_bbox = (l[0] + r[0])/2, (l[1] + r[1])/2, l[2], r[3]
    brain_coo = linalg_utils.bbox_to_coords(*brain_bbox)
    angle = - meta['rotation']  # the 'rotation' was intended for rotation of the image
    center = image_shape[1] / 2 - 1, image_shape[0] / 2 - 1  # order should be as in coords
    brain_coo = linalg_utils.rotate_coords(brain_coo, center, angle, degrees=True)
    return brain_coo


def make_brain_mask(ontology_info, brain_coo, image_shape) -> 'np.ndarray[bool]':
    mask_path = ontology_info.mask_path(mask_name='Root')
    mask = ontology_info.open_mask(mask_path)
    mask |= np.flip(mask, axis=1)
    src_coo = linalg_utils.bbox_to_coords(0, mask.shape[0], 0, mask.shape[1])
    pad = ((0, image_shape[0]-mask.shape[0]), (0, image_shape[1]-mask.shape[1]))
    mask = np.pad(mask, pad, constant_values=0)
    tform = skt.estimate_transform('affine', brain_coo, src_coo)
    mask = skt.warp(mask.astype(np.float), tform, order=3, preserve_range=True)
    return mask > 0.5


def make_head_mask(image, brain_mask):
    x = skf.gaussian(image, sigma=2, truncate=2)
    x = np.where(x < 0.5, 1, 0)
    c = center_of_binary_image(brain_mask)
    mask = sks.flood(x, c)
    return mask & ~brain_mask


def center_of_binary_image(bimg: 'np.ndarray[bool]') -> (int, int):
    rows = np.any(bimg, axis=1)
    cols = np.any(bimg, axis=0)
    u = np.where(rows)[0][[0, -1]].sum() // 2
    v = np.where(cols)[0][[0, -1]].sum() // 2
    return u, v
