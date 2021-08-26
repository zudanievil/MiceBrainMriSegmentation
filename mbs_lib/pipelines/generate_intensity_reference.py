import tqdm
import yaml
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd
import skimage.exposure as ske
import scipy.optimize as sco
import skimage.segmentation as sks
from ..core import info_classes
from . import segment_head_and_brain_subpipeline


def main(srfi: info_classes.segmentation_result_folder_info_like):
    generate_reference_table(srfi)
    plot_reference_table(srfi)
    add_references_to_metadata(srfi)


# ======================================================================
def generate_reference_table(srfi: info_classes.segmentation_result_folder_info_like):
    """
    first, obtains mask of brain and head boolean masks
    for the image using `pipelines.segment_head_and_brain_subpipeline.main` function
    then computes different functions over these masks
    (see `pipelines.generate_intensity_reference.get_metrics_over_masks`).
    saves table to "additional_references/ref_table.txt" file in results folder.
    """
    save_folder = srfi.folder() / 'additional_references'
    save_folder.mkdir(exist_ok=True)
    image_folder_info = srfi.image_folder_info()
    ontology_folder_info = srfi.ontology_folder_info()
    fname_fields = image_folder_info.configuration()['file_name_fields']
    stats = []
    pb = tqdm.tqdm(leave=False, total=len(image_folder_info))
    pb.set_description('reference_table_generation')
    for image_info in image_folder_info:
        pb.update()
        pb.set_postfix_str(image_info.name())
        chuncks = image_info.name().split('_')
        chuncks = {f: int(c) if c.isnumeric() else c for f, c in zip(fname_fields, chuncks)}
        chuncks['name'] = image_info.name()
        ontology_info = ontology_folder_info.ontology_info(chuncks['frame'])
        brain_mask, head_mask = segment_head_and_brain_subpipeline.main(image_info, ontology_info)
        img = image_info.image()
        chuncks.update(get_metrics_over_masks(brain_mask, head_mask, img))
        stats.append(chuncks)
    pb.close()
    stats = pd.DataFrame(stats)
    stats.set_index(['name'], inplace=True)
    stats.to_csv(save_folder/'ref_table.txt', sep='\t', index=True)


def get_metrics_over_masks(brain_mask: "np.ndarray[bool]", head_mask: "np.ndarray[bool]", img: np.ndarray):
    """
    filters desired regions of the image using `MASKS` constant,
    then computes different metrics over the masked image using `METRICS` constant
    """
    img = img.astype(float)  # for some reason ndi functions over int images give weird results
    result = {}
    for mask_name in MASKS:
        values = MASKS[mask_name](img, brain_mask, head_mask)
        for metric_name in METRICS:
            column = mask_name + ' ' + metric_name
            result[column] = METRICS[metric_name](values)
    result['histogram bg peak'] = mri_bg_peak(img)  # no reason to use it with masks
    return result


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


def mri_bg_peak(img: np.ndarray) -> float:
    """
    This function fits a gaussian onto image intensity histogram peak (the one, corresponding to background).
    this may be a better normalization metric than mean or standard deviation.
    This function, unlike others, is 'aware' of intensity distribution specifics.
    :returns: peak position (background intensity)
    """
    img_truncated = img[img < 0.3*np.max(img)].astype(float)  # we truncate the long tail of high values
    hist_y, hist_x = ske.histogram(img_truncated, nbins=200)
    peak_idx = hist_y[2:].argmax() + 2  # sometimes 1st bin values 'outweigh' the gaussian
    fit_params = fit_gaussian_on_histogram_peak(hist_x, hist_y, peak_idx)
    return fit_params[1]  # fit_params == (a, mu, sigma)


def fit_gaussian_on_histogram_peak(x: "np.ndarray[float]", y: "np.ndarray[float]", peak_idx: int):
    """
    procedure of fitting scaled gaussian with least-squares optimizer.
    :param peak_idx: index of y near the peak that is being interpolated.
    """
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


def gaussian(x: "np.ndarray[float]", a: float, mu: float, sigma: float):
    """
    computes gaussian PDF
    :returns: a*exp(  -({x-mu} /sigma)^2  )
    """
    return a * np.exp(- ((x - mu) / sigma) ** 2)


# ==================================================================
def plot_reference_table(srfi: info_classes.segmentation_result_folder_info_like):
    """
    plots the values from the reference table, saves them to the same folder
    """
    save_folder = srfi.folder() / 'additional_references'
    image_folder_info = srfi.image_folder_info()
    fname_fields = image_folder_info.configuration()['file_name_fields']
    t = pd.read_csv(save_folder/'ref_table.txt', sep='\t', index_col=0)
    batch_spec = srfi.configuration()['batching']
    metrics = set(t.columns)
    metrics.difference_update(set(fname_fields))

    plt.ioff()
    pb = tqdm.tqdm(leave=False, total=len(metrics))
    pb.set_description('plotting reference table')
    for metric in metrics:
        pb.update()
        pb.set_postfix_str(metric)

        t2 = t[[metric, *batch_spec['compare_by'], *batch_spec['match_by']]]
        t2 = t2.groupby(by=[*batch_spec['match_by'], *batch_spec['compare_by']]).mean()
        t2 = t2.unstack(batch_spec['compare_by'][0])

        plt.figure(figsize=(12, 10))
        x_values = t2.columns.get_level_values(batch_spec['compare_by'][0]).to_numpy()
        for idx in t2.index:
            metric_value = t2.loc[idx].to_numpy()
            plt.plot(x_values, metric_value, label=idx)
        plt.grid()
        plt.legend()
        plt.title(metric)
        plt.savefig(save_folder / (metric + '.png'), dpi=96, format='png')
        plt.close()
    pb.close()
    plt.ion()


# =======================================================================
def add_references_to_metadata(srfi: info_classes.segmentation_result_folder_info_like):
    """
    Adds values from reference table to corresponding metadata files.
    This is a rather bad decision, we'll probably fix this in the future.
    """
    save_folder = srfi.folder() / 'additional_references'
    image_folder_info = srfi.image_folder_info()
    fname_fields = image_folder_info.configuration()['file_name_fields']
    t = pd.read_csv(save_folder / 'ref_table.txt', sep='\t', index_col=0)
    t.drop(columns=fname_fields, inplace=True)
    pb = tqdm.tqdm(leave=False, total=len(image_folder_info))
    pb.set_description('reference_table_generation')
    for name in t.index:
        pb.update()
        pb.set_postfix_str(name)
        image_info = image_folder_info.image_info(name)
        meta = image_info.metadata()
        ref = t.loc[name].to_dict()
        ref = {'ER_' + k.replace(' ', '_'): ref[k] for k in ref}
        meta.update(ref)
        with image_info.metadata_path().open('wt') as f:
            yaml.safe_dump(meta, f)
    pb.close()
