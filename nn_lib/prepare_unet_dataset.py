#! /usr/bin/env python3

import pathlib
import yaml
import numpy as np
import PIL.Image
import skimage.transform
import skimage.segmentation
import skimage.filters
import skimage.feature

__all__ = ['prepare_dataset', 'split_dataset', 'rotate_yx', 'sigmoid_with_quantiles']


def _load_meta(p: pathlib.Path) -> dict:
    with p.open('rt') as f:
        meta = yaml.safe_load(f.read())
    return meta


def _load_image(p: pathlib.Path) -> np.ndarray:
    image = np.load(p, fix_imports=False)
    return image


def _load_whole_brain_mask(masks_folder: pathlib.Path, meta: dict) -> np.ndarray:
    mask_path = masks_folder / meta['frame'] / 'root' / 'Root.png'
    mask = PIL.Image.open(mask_path)
    mask = np.array(mask.getdata(), dtype=np.uint8).reshape((mask.size[1], mask.size[0]))
    mask = mask[:, mask.shape[1]//2:] > mask.max()//2
    return mask


def _mask_brain(meta, img_shape, brain_mask) -> np.ndarray:
    img_shape = np.array(img_shape, dtype=np.int)
    result_mask = np.zeros(img_shape)
    for side in ['l', 'r']:
        bx = meta[f'{side}bbox']
        bx_shape = np.array([bx[1] - bx[0], bx[3] - bx[2]], dtype=np.int)
        mask = skimage.transform.resize(brain_mask, bx_shape, preserve_range=True)
        if side == 'l':
            mask = np.flip(mask, axis=1)
        result_mask[bx[0]:bx[1], bx[2]:bx[3]] = mask
    result_mask = skimage.transform.rotate(result_mask, meta['rotation'], preserve_range=True) >0.5
    return result_mask  # bool


def _segment_out_water_pipe(image):
    e = skimage.feature.canny(image, sigma=3, low_threshold=0.2, high_threshold=0.8)
    e2 = np.zeros(e.shape, dtype=bool)
    e2[:e.shape[0]//2, e.shape[1]//2:] = e[:e.shape[0]//2, e.shape[1]//2:]
    ws = skimage.segmentation.watershed(e2*1)
    labels = np.unique(ws)
    areas = []
    for l in labels:
        areas.append((ws == l).sum())
    areas = np.array(areas)
    without_stupid_water_pipe = ws == labels[np.argmax(areas)]
    without_stupid_water_pipe &= (image > 0.5)
    return without_stupid_water_pipe  # bool


def sigmoid_with_quantiles(img: np.ndarray, sigmoid_gain: int = 10, sigmoid_cutoff: float = 0.5,
                           quantiles: (float, float) = (0.07, 0.93)) -> np.ndarray:
    flat = img.flatten()
    idx = np.argsort(flat)
    low = flat[idx[int(quantiles[0] * len(idx))]]
    high = flat[idx[int(quantiles[1] * len(idx))]]
    img = (img - low) / (high - low)
    img = 1 / (1 + np.exp(sigmoid_gain * (sigmoid_cutoff - img)))
    return img


def rotate_yx(yx, center, rotation_degs):
    yx = yx - center
    angles = np.arctan2(yx[..., 0], yx[..., 1])
    radii = np.sqrt(np.sum(yx ** 2, axis=-1))
    angles += rotation_degs / 180 * np.pi
    yx = np.stack((radii * np.sin(angles), radii * np.cos(angles)), axis=-1)
    yx += center
    return yx


def prepare_dataset(project: pathlib.Path, masks_folder: pathlib.Path, dataset_folder: pathlib.Path) -> None:
    for meta_path in (project / 'meta').iterdir():
        name = meta_path.stem
        image_path = project / 'img' / (name + '.npy')
        meta = _load_meta(meta_path)
        image = _load_image(image_path)
        image = sigmoid_with_quantiles(image)
        brain_mask = _load_whole_brain_mask(masks_folder, meta)
        brain_mask = _mask_brain(meta, image.shape, brain_mask)
        no_water_pipe = _segment_out_water_pipe(image)
        head_mask = no_water_pipe & (~ brain_mask) & (image > 0.5)
        gnd = np.stack([brain_mask, head_mask], axis=-1)*1.0
        image = image[..., np.newaxis]
        data = {'inp': image.astype(np.float32), 'gnd': gnd.astype(np.float32)}
        data_path = dataset_folder / (name + '.npz')
        np.savez(data_path, **data)
        print(data_path.as_posix())
    print('Done: ', project.as_posix())


def split_dataset(dataset_folder: pathlib.Path, test_ratio: float = 0.02):
    print('splitting dataset...', end='\t')
    paths = []
    for path in dataset_folder.iterdir():
        if path.suffix == '.npz':
            paths.append(path)
    paths = np.array(paths)
    np.random.shuffle(paths)
    split = int(len(paths)*test_ratio)
    test_paths = paths[:split]
    train_paths = paths[split:]
    test_folder = dataset_folder / 'test'
    train_folder = dataset_folder / 'train'
    test_folder.mkdir(exist_ok=True)
    train_folder.mkdir(exist_ok=True)
    for path in test_paths:
        try:
            path.rename(test_folder / path.name)
        except FileExistsError:
            path.unlink()
    for path in train_paths:
        try:
            path.rename(train_folder / path.name)
        except FileExistsError:
            path.unlink()
    print('done')


if __name__ == '__main__':
    cwd = pathlib.Path('c:/users/user/desktop/new_segm')
    dataset_folder = cwd / 'brain_and_head_dataset'
    for g in ['2.4DNP', '2DG', 'saline']:
        project = cwd / g
        masks_folder = cwd / 'masks_01'
        prepare_dataset(project, masks_folder, dataset_folder)
        split_dataset(dataset_folder)