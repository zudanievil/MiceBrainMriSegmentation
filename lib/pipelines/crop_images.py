import numpy
import pathlib
import matplotlib.pyplot
import skimage.transform

from ..core import info_classes
from ..utils import linalg_utils


_LOCALS = {
    'skimage_transformations_kwargs': {
        'order': 3,
        'preserve_range': True,
    },
}


def crop_rotate_image(image: numpy.ndarray, meta: dict, frame_shapes: dict) -> numpy.ndarray:
    tform_kwargs = _LOCALS['skimage_transformations_kwargs']
    shape = frame_shapes[meta['frame']]
    shape = (shape[0], shape[1] // 2)
    image = skimage.transform.rotate(
        image, -meta['rotation'], **tform_kwargs)
    left = linalg_utils.bbox_crop(image, meta['lbbox'])
    left = skimage.transform.resize(
        left, shape, **tform_kwargs)
    right = linalg_utils.bbox_crop(image, meta['rbbox'])
    right = skimage.transform.resize(
        right, shape, **tform_kwargs)
    image = numpy.concatenate([left, right], axis=1)
    return image


def main(image_folder_info: info_classes.image_folder_info_like, save_png_previews: bool = True):
    image_folder_info = info_classes.ImageFolderInfo(image_folder_info)
    frame_shapes = image_folder_info.specification()['cropped_image_shapes']
    for image_info in image_folder_info:
        print(image_info.image_path())
        image = image_info.image()
        meta = image_info.metadata()
        image = crop_rotate_image(image, meta, frame_shapes)

        p = image_info.cropped_image_path()
        numpy.save(p, image, fix_imports=False)
        if save_png_previews:
            matplotlib.pyplot.imsave(p.with_suffix('.png'), image, cmap='gray', format='png')
