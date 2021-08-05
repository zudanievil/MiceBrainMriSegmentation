import numpy
import tqdm
import matplotlib.pyplot
import skimage.transform

from ..core import info_classes
from ..utils import linalg_utils


def crop_rotate_image(
        image: numpy.ndarray,
        meta: dict,
        frame_shapes: dict,
        tform_kwargs: dict,
        ) -> numpy.ndarray:
    """
    :param image: grayscale image
    uses metadata entries to transform the image:
    first rotates the image to make hemisphere junction vertical
    then selects hemispheres (with help of bounding boxes)
    then resizes them depending on frame
    and finally concatenates resized halves.
    :returns: grayscale image of brain
    """
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


def main(image_folder_info: info_classes.image_folder_info_like, save_png_previews: bool = True) -> None:
    """
    crops brain images and rescales them to specified shapes (see `cropped_image_shapes` in image folder configuration)
    resizing is hard-coded to depend on `frame` metadata entry, since `frame` is essentially
    a named brain section coordinate, and brain size varies with frame only.
    each hemisphere is resized individually, which means that left half of the image contains only left hemisphere
    and right half of the image is precisely right hemisphere.
    :param save_png_previews: saves a black and white .png preview of .npy file for visual inspection, debugging
    """
    image_folder_info = info_classes.ImageFolderInfo(image_folder_info)
    frame_shapes = image_folder_info.specification()["cropped_image_shapes"]
    tform_kwargs = image_folder_info.specification()["image_transform_interpolation"]

    progress_bar = tqdm.tqdm(leave=False, total=len(image_folder_info))
    for image_info in image_folder_info:
        progress_bar.set_postfix_str(image_info.name())
        progress_bar.update()

        image = image_info.image()
        meta = image_info.metadata()
        image = crop_rotate_image(image, meta, frame_shapes, tform_kwargs)

        p = image_info.cropped_image_path()
        numpy.save(p, image, fix_imports=False)
        if save_png_previews:
            matplotlib.pyplot.imsave(p.with_suffix('.png'), image, cmap='gray', format='png')

    progress_bar.close()
