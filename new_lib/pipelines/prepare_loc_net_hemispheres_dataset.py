import yaml
import pathlib
import numpy as np

from ..core import info_classes
from ..utils import linalg_utils

__all__ = ['main']


def main(dataset_folder: 'pathlib.Path or str',
         image_dict_path: 'pathlib.Path or str' = None,
         project_map_path: 'pathlib.Path or str' = None):
    """
    only one of image_dict path or
    project_map_path must be specified
    :param image_dict_path: create a dataset from images related to different projects
    images and projects are specified in the 'image_dict.yml' like:
        path/to/project0.yaml:
            - image0_name
            - image1_name
        path/to/project1.yaml:
            - image2_name
            - image3_name
    :param project_map_path: create a dataset from all the images in a project
    that have metadata
    :return: writes to {dataset_folder}/new_images
    files with name image_name.npz
    each file has contrasted image under key 'inp',
    and affine transformations that bring each hemisphere to the cener-left and center-right parts of the
    image under keys 'gnd_l' and 'gnd_r'
    """
    dataset_folder = pathlib.Path(dataset_folder)
    dataset_folder /= 'new_images'
    dataset_folder.mkdir(parents=True, exist_ok=True)
    if image_dict_path and not project_map_path:
        info_gen = generator_from_dict_file(pathlib.Path(image_dict_path))
    elif project_map_path and not image_dict_path:
        info_gen = generator_from_project_map(pathlib.Path(project_map_path))
    else:
        raise AssertionError("one and only one of 'image_dict_path' and 'project_path' must be specified")
    for image_info in info_gen:
        print(image_info.name)
        image = image_info.image()
        image = linalg_utils.sigmoid_with_quantiles(image)
        l_coo, r_coo = get_bboxes_as_coords(image_info)
        target_l_coo, target_r_coo = get_target_coords(image_info, image.shape)
        gnd_l = linalg_utils.estimate_affine(src_coo=l_coo, dst_coo=target_l_coo)
        gnd_r = linalg_utils.estimate_affine(src_coo=r_coo, dst_coo=target_r_coo)
        image = linalg_utils.image_bhwc_to_bcwh(image)
        np.savez(dataset_folder / image_info.name, inp=image.astype(np.float32),
                 gnd_r=gnd_r.astype(np.float32), gnd_l=gnd_l.astype(np.float32))


def generator_from_dict_file(image_dict_path):
    with image_dict_path.open('rt') as f:
        image_dict = yaml.safe_load(f.read)
    for project in image_dict:
        project_info = info_classes.ProjectInfo()
        for image_name in image_dict[project]:
            yield info_classes.ImageInfo(project_info, image_name)


def generator_from_project_map(project_path):
    pi = info_classes.ProjectInfo.read(project_path)
    for image_path in pi.image_folder.iterdir():
        image_name = image_path.with_suffix('').name
        image_info = info_classes.ImageInfo(pi, image_name)
        if image_info.meta_path().exists() and \
                image_info.cropped_image_path().exists():
            yield image_info


def get_bboxes_as_coords(image_info: info_classes.ImageInfo) -> (np.ndarray, np.ndarray):
    meta = image_info.meta()
    l = linalg_utils.bbox_to_coords(*meta['lbbox'])
    r = linalg_utils.bbox_to_coords(*meta['rbbox'])
    angle = - meta['rotation']  # the 'rotation' was intended for rotation of the image
    l = linalg_utils.rotate_coords(l, (128, 128), angle, degrees=True)
    r = linalg_utils.rotate_coords(r, (128, 128), angle, degrees=True)
    return l, r


def get_target_coords(image_info: info_classes.ImageInfo, image_shape) -> (np.ndarray, np.ndarray):
    cy = image_shape[0]/2. - 1
    cx = image_shape[1]/2. - 1
    dy, dx = image_info.cropped_image().shape
    dy /= 2.
    dx /= 2.
    l = linalg_utils.bbox_to_coords(y0=cy-dy, y1=cy+dy, x0=cx-dx, x1=cx)
    r = linalg_utils.bbox_to_coords(y0=cy-dy, y1=cy+dy, x0=cx, x1=cx+dx)
    return l, r
