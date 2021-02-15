"""
Utility functions to prepare datasets.
Usage:
>>> dataset_folder = pathlib.Path('c:/users/user/desktop/unet_dataset')
>>> image_dict_path = pathlib.Path('c:/users/user/desktop/image_dict.yml')
>>> create_dataset_folder(dataset_folder)
>>> dataset_from_image_dict(image_dict_path, dataset_folder)
"""

import dataclasses
import datetime
import pathlib

import PIL.Image
import numpy as np
import skimage.feature
import skimage.filters
import skimage.segmentation
import skimage.transform
import yaml

__all__ = ['dataset_from_image_dict', 'create_dataset_folder']


def create_dataset_folder(path: pathlib.Path):
    paths = (path, path / 'train', path / 'test', path / 'new_images')
    for p in paths:
        p.mkdir(exist_ok=False, parents=True)


def dataset_from_image_dict(image_dict_path: pathlib.Path, dataset_folder: pathlib.Path) -> None:
    """
    Main function of the module
    Prepares images for segmentation. uses uncropped images + image metadata + brain mask from masks_folder.
    Computes brain mask with the aid of a prerendered one + metadata.
    Segments the head out based on contrast/borders.
    Does not run checks on file existence/redundancy.
    :param image_dict_path: yaml file of structure:
    ---
    path/to/brain/mask0:
        - path/to/image0
        - path/to/image1
    path/to/brain/mask1:
        - path/to/image2
        - path/to/image3
    ---
    paths must be absolute.
    :return: puts prepared dataset elements into {dataset_folder}/new_images directory.
    Takes approximately 50 ms per image.
    """
    dataset_folder = dataset_folder / 'new_images'
    for image_info in image_info_generator(image_dict_path):
        gnd, image = load_and_segment(image_info)
        save_path = dataset_folder / image_info.name()
        np.savez(save_path, inp=image, gnd=gnd)
        print(datetime.datetime.now(), image_info, 'done')
    print(datetime.datetime.now(), 'finished')


def load_and_segment(image_info: 'ImageInfo') -> (np.ndarray, np.ndarray):
    image_contrasted = image_info.image_contrasted()
    brain_mask = mask_the_brain(image_info, image_contrasted.shape)
    head_mask = mask_head(image_contrasted, brain_mask)
    head_mask = head_mask & (~ brain_mask)
    image_contrasted = image_contrasted[..., np.newaxis].astype(np.float32)
    segmentation = np.stack([brain_mask, head_mask], axis=-1).astype(np.float32)
    return segmentation, image_contrasted


@dataclasses.dataclass(frozen=True)
class ImageInfo(object):
    """
    Small utility class for convenient data retrieval.
    Note that it reads image data etc from disk, and does no caching
    """
    mask_path: pathlib.Path
    image_path: pathlib.Path

    def meta(self) -> dict:
        name = self.image_path.with_suffix('.yml').name
        folder = self.image_path.parent.parent / 'meta'
        path = folder / name
        with path.open('rt') as f:
            meta = yaml.safe_load(f.read())
        return meta

    def image_contrasted(self) -> 'np.ndarray[float]':
        return sigmoid_with_quantiles(self.image())

    def image(self) -> np.ndarray:
        return np.load(self.image_path, fix_imports=False)

    def brain_mask_right_half(self) -> 'np.ndarray[bool]':
        """
        :returns thresholded (>= 1/2 of max value) right half of the mask
        """
        mask = PIL.Image.open(self.mask_path)
        mask = np.array(mask.getdata(), dtype=np.uint8).reshape((mask.size[1], mask.size[0]))
        mask = mask[:, mask.shape[1] // 2:] >= mask.max() // 2
        return mask

    def masks_folder(self) -> pathlib.Path:  # TODO: fix this
        raise NotImplementedError
        return self.mask_path.parent.parent

    def frame_id(self) -> str:  # TODO: fix this
        raise NotImplementedError
        return self.mask_path.parent.name

    def name(self):
        return self.image_path.with_suffix('').name


def image_info_generator(image_dict_file) -> 'collections.Iterable[ImageInfo]':
    with image_dict_file.open('rt') as f:
        image_dict = yaml.safe_load(f.read())
    for mask_path in image_dict:
        image_list = image_dict[mask_path]
        for image_path in image_list:
            yield ImageInfo(pathlib.Path(mask_path), pathlib.Path(image_path))


def mask_the_brain(image_info: ImageInfo, image_shape: (int, int)) -> 'np.ndarray[bool]':
    meta = image_info.meta()
    brain_mask = image_info.brain_mask_right_half()
    result = np.zeros(image_shape, dtype=np.float)
    for side in ['l', 'r']:
        bx = meta[f'{side}bbox']
        bx_shape = np.array([bx[1] - bx[0], bx[3] - bx[2]], dtype=np.int)
        mask = skimage.transform.resize(brain_mask*1.0, bx_shape)
        # do not resize boolean images, this is unstable!!!
        # from time to time the resized image is blank (especially when < 100x100 px)
        if side == 'l':
            mask = np.flip(mask, axis=1)
        result[bx[0]:bx[1], bx[2]:bx[3]] = mask
    result = skimage.transform.rotate(result, meta['rotation']) > 0.5
    return result


def mask_head(image, brain_mask):
    def center_of(bimg):
        rows = np.any(bimg, axis=1)
        cols = np.any(bimg, axis=0)
        u = np.where(rows)[0][[0, -1]].sum() // 2
        v = np.where(cols)[0][[0, -1]].sum() // 2
        return u, v

    x = skimage.filters.gaussian(image, sigma=2, truncate=2)
    x = np.where(x < 0.5, 1, 0)
    c = center_of(brain_mask)
    mask = skimage.segmentation.flood(x, c)
    return mask


# TODO: probably move to transform utils, rename
def sigmoid_with_quantiles(img: np.ndarray, sigmoid_gain: int = 10, sigmoid_cutoff: float = 0.5,
                           quantiles: (float, float) = (0.07, 0.93)) -> np.ndarray:
    flat = img.flatten()
    idx = np.argsort(flat)
    low = flat[idx[int(quantiles[0] * len(idx))]]
    high = flat[idx[int(quantiles[1] * len(idx))]]
    img = (img - low) / (high - low)
    img = 1 / (1 + np.exp(sigmoid_gain * (sigmoid_cutoff - img)))
    return img


# TODO: definitely move to transform utils, this has no use here
def rotate_yx_coordinates(yx, center, rotation_degs):
    yx = yx - center
    angles = np.arctan2(yx[..., 0], yx[..., 1])
    radii = np.sqrt(np.sum(yx ** 2, axis=-1))
    angles += rotation_degs / 180 * np.pi
    yx = np.stack((radii * np.sin(angles), radii * np.cos(angles)), axis=-1)
    yx += center
    return yx
