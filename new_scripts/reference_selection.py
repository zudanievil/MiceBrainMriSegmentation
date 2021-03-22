import sys
import pathlib
import yaml
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd
import skimage.exposure as ske
import scipy.optimize as sco
import skimage.segmentation as sks
sys.path.append('..')
from ignore_nn_lib import dataset_preparation


def gaussian(x, a, mu, sigma):
    return a * np.exp(- ((x - mu) / sigma) ** 2)


def fit_gaussian_on_histogram_peak(x, y, peak_idx):
    # get init estimates
    a_init = y[peak_idx]
    mu_init = x[peak_idx]
    gthm = y > (a_init / 2)  # gthm -- greater than half maximum
    gthm = np.where(sks.flood(gthm.astype(int), peak_idx))[0]  # for some reason does not work with bools
    min_hm_x, max_hm_x = x[gthm[[0, -1]]]
    sigma_init = (max_hm_x - min_hm_x)/2.355  # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    init_params = np.array([a_init, mu_init, sigma_init])
    # select 2*sigma range
    idx = np.where((x > mu_init - sigma_init*2) & (x < mu_init+sigma_init*2))
    x = x[idx]
    y = y[idx]
    # fit
    err = lambda params, x, y: y - gaussian(x, *params)
    fit_params, _ = sco.leastsq(err, init_params, args=(x, y))
    return fit_params


def mri_bg_peak(img):
    # this function, unlike others, is 'aware' of data specifics.
    img_truncated = img[img < 0.3*np.max(img)].astype(float)  # we truncate the long tail of high values
    hist_y, hist_x = ske.histogram(img_truncated, nbins=200)
    peak_idx = hist_y[2:].argmax() + 2  # sometimes 1st bin values 'outweigh' the gaussian
    fit_params = fit_gaussian_on_histogram_peak(hist_x, hist_y, peak_idx)
    return fit_params[1]  # fit_params == (a, mu, sigma)


def get_metrics_over_masks(segmentation, img):
    img = img.astype(float)  # for some reason ndi functions over int images give wierd results
    brain = segmentation[..., 0]
    head = segmentation[..., 1]
    result = {}
    for mask_name in MASKS:
        values = MASKS[mask_name](img, brain, head)
        for metric_name in METRICS:
            column = mask_name + ' ' + metric_name
            result[column] = METRICS[metric_name](values)
    result['histogram bg peak'] = mri_bg_peak(img)  # no reason to use it with masks
    return result


def plot_metrics(kwargs):
    t = pd.read_csv(kwargs['stat_path'], sep='\t')
    metrics = set(t.columns)
    metrics.difference_update({'name', 'hour', 'animal', 'frame'})
    for column in metrics:
        plot_name = kwargs['exp'] + ' ' + column
        print(plot_name)
        t2 = t[[column, 'hour', 'frame']]
        t2 = t2.groupby(by=['frame', 'hour']).mean().unstack('hour')

        plt.figure(figsize=(12, 10))
        hour = t2.columns.get_level_values(1).to_numpy().astype(float)
        for frame in t2.index:
            metric_value = t2.loc[frame].to_numpy()
            plt.plot(hour, metric_value, label=frame)
        plt.grid()
        plt.legend()
        plt.title(plot_name)
        plt.savefig(kwargs['save_folder'] / (plot_name + '.png'), dpi=96, format='png')
        plt.close()


def make_metric_table(kwargs):
    stats = []
    for img_path in kwargs['img_folder'].iterdir():
        print(img_path.name)
        _, hour, animal, frame = img_path.with_suffix('').name.split('_')
        hour = int(hour)
        animal = int(animal)
        mask_path = kwargs['masks_folder'] / frame / 'root' / 'Root.png'

        img_info = dataset_preparation.ImageInfo(mask_path, img_path)
        segmentation, _ = dataset_preparation.load_and_segment(img_info)
        segmentation = segmentation > 0
        img = img_info.image()
        stat = get_metrics_over_masks(segmentation, img)
        del segmentation, img

        stat['name'] = img_info.name()
        stat['hour'] = hour
        stat['animal'] = animal
        stat['frame'] = frame
        stats.append(stat)
    stats = pd.DataFrame(stats)
    stats.set_index(['name'], inplace=True)
    stats.to_csv(kwargs['stat_path'], sep='\t')


def add_metrics_to_metas(meta_folder, meta_save_folder, table_path):
    def load_meta(path):
        with path.open('rt') as f:
            meta = yaml.safe_load(f.read())
        return meta

    def save_meta(meta, path):
        with path.open('wt') as f:
            yaml.safe_dump(meta, f)

    def load_names_and_new_refs(path):
        t = pd.read_csv(path, sep='\t', index_col='name')
        t.drop(columns=['hour', 'animal', 'frame'], inplace=True)
        return t.index, t

    assert meta_folder != meta_save_folder
    names, refs = load_names_and_new_refs(table_path)
    for name in names:
        print(name)
        meta_path = meta_folder / (name + '.yml')
        meta_save_path = meta_save_folder / (name + '.yml')
        meta = load_meta(meta_path)
        ref = refs.loc[name].to_dict()
        ref = {'ER_' + k.replace(' ', '_'): ref[k] for k in ref}
        meta.update(ref)
        save_meta(meta, meta_save_path)





METRICS = {
    'mean': lambda x: ndi.mean(x),
    'median': lambda x: ndi.median(x),
    'std': lambda x: ndi.standard_deviation(x),
    }

MASKS = {
    'brain': lambda i, b, h: i[b],
    'head': lambda i, b, h: i[h],
    'animal': lambda i, b, h: i[b | h],
    'image': lambda i, b, h: i.flatten(),
    'back': lambda i, b, h: i[~(b | h)],
    }

if __name__ == '__main__':
    for exp in ('c57bl', 'cgtg'):
        kwargs = {
            'exp': exp,

            'img_folder': pathlib.Path(f"C:\\Users\\user\\files\\lab\\CgTg\\{exp}\\img"),
            'masks_folder': pathlib.Path("C:\\Users\\user\\files\\lab\\masks_actual"),

            'save_folder': pathlib.Path(f"C:\\Users\\user\\desktop\\ref_metric_plots"),
            'stat_path': pathlib.Path(f"C:\\Users\\user\\desktop\\ref_metric_plots") / (exp + '_metrics.txt'),
            }
        # print('TABELING')
        # make_metric_table(kwargs)
        # print('PLOTTING')
        # plot_metrics(kwargs)

        #######
        meta_folder = pathlib.Path(f"C:\\Users\\user\\files\\lab\\CgTg\\{exp}\\meta_orig")
        meta_save_folder = pathlib.Path(f"C:\\Users\\user\\files\\lab\\CgTg\\{exp}\\meta")
        table_path = pathlib.Path(f"C:\\Users\\user\\desktop\\ref_metric_plots\\{exp}_metrics.txt")
        add_metrics_to_metas(meta_folder, meta_save_folder, table_path)







