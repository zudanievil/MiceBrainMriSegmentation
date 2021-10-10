import importlib
import pathlib
import sys
import typing
import tqdm
import numpy
import pandas
import scipy.stats
import skimage.filters
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ..core import info_classes
from ..utils import miscellaneous_utils


PLOT_FOLDER_MAX_LENGTH = 100


def get_plot_folder(srfi: info_classes.segmentation_result_folder_info_like) -> pathlib.Path:
    """wraps code for getting plot folder and validating it's length"""
    plot_folder = srfi.plot_folder()
    assert len(str(plot_folder)) <= PLOT_FOLDER_MAX_LENGTH, \
        f'plot folder path length <= {PLOT_FOLDER_MAX_LENGTH} is recommended'
    plot_folder.mkdir(exist_ok=True)
    return plot_folder


def load_mask_permutation(srfi: info_classes.segmentation_result_folder_info_like) -> callable:
    """unsafe function reads code from "mask_permutation.py" file in results folder """
    p = srfi.mask_permutation_path()
    if p.exists():
        sys.path.insert(0, str(p.parent))  # this may not be thread-safe
        module = importlib.import_module(p.stem)
        sys.path.pop(0)
        return module.mask_permutation
    else:
        return lambda x, *y: x


class UniversumSet:
    @classmethod
    def __contains__(cls, item):
        return True


def get_selected_structures(p: pathlib.Path) -> typing.Union[typing.Set[str], UniversumSet]:
    if not p.exists():
        return UniversumSet()
    else:
        with p.open("rt") as f:
            ss = {name.strip() for name in f.readlines()}
        return ss


def batches_gen(srfi: info_classes.segmentation_result_folder_info_like, batch_range: slice = None) -> \
        typing.Iterator[typing.Tuple[int, "numpy.ndarray[bool]", "numpy.ndarray[str]"]]:
    """
    first time yields batches length,
    then yields batch number, reference group mask (boolean numpy array),
    image names (numpy array for strings).
    """
    path = srfi.batches_path()
    batches = pandas.read_csv(path, sep='\t', index_col=0, header=[0, 1], na_values='NA',
                              true_values='True', false_values='False')
    ref_mask = batches.columns.get_level_values(0).to_numpy()
    ref_mask = ref_mask == 'True'
    batches = batches.loc[batch_range] if batch_range else batches
    yield len(batches)
    for i in batches.index:
        yield i, ref_mask, batches.loc[i].to_numpy().astype(str)


def get_imgs_metas(
        sfri: info_classes.segmentation_result_folder_info_like,
        batch: "numpy.ndarray[str]"
        ) -> typing.Tuple[numpy.ndarray, pandas.DataFrame]:
    """loads batch of images (numpy array) and metas (DataFrame)"""
    ifi = sfri.image_folder_info()
    metas = []
    imgs = []
    for name in batch:
        if name == 'nan':
            continue
        ii = ifi.image_info(name)
        metas.append(ii.metadata())
        imgs.append(ii.cropped_image())
    return numpy.stack(imgs, axis=0), pandas.DataFrame(metas)


def normalize_imgs(imgs, metas, spec) -> numpy.ndarray:
    """normalizes images to account for intensity shift"""
    base_key = spec["normalization"]["subtract"]
    var_key = spec["normalization"]["divide_by"]
    hm = metas[base_key].to_numpy()[..., numpy.newaxis, numpy.newaxis] if base_key else 0
    std = metas[var_key].to_numpy()[..., numpy.newaxis, numpy.newaxis] if var_key else 1
    imgs = (imgs - hm) / std
    return imgs


def calculate_pixwise_difference_of_means(imgs, ref_mask) -> numpy.ndarray:
    """difference of means between "reference" and "treated" groups"""
    return imgs[~ ref_mask].mean(axis=0) - imgs[ref_mask].mean(axis=0)


def calculate_pixwise_mean_of_difference(imgs, ref_mask) -> numpy.ndarray:
    """calculates difference of means between "reference" and "treated" groups. for repeated measurements"""
    return (imgs[~ ref_mask] - imgs[ref_mask]).mean(axis=0)


def calculate_pixwise_t_ind(imgs, ref_mask) -> (numpy.ndarray, numpy.ndarray):
    """
    performs independent sample t-test between "reference" and "treated" groups.,
    returns: test_statistic, p_values
    """
    return scipy.stats.ttest_ind(imgs[~ ref_mask], imgs[ref_mask], axis=0, equal_var=False)


def calculate_pixwise_t_rel(imgs, ref_mask) -> (numpy.ndarray, numpy.ndarray):
    """
    performs related sample t-test between "reference" and "treated" groups.
    returns: test_statistic, p_values
    """
    return scipy.stats.ttest_rel(imgs[~ ref_mask], imgs[ref_mask], axis=0)


def smooth_out_p_values(pval, gaussian_kwargs) -> numpy.ndarray:
    """
    smooths out p-values with gaussian:
    p' = 1 - G(1 - p), where p is p-values, G is 2d gaussian filter
    returns: p_values
    """
    return 1 - skimage.filters.gaussian(1 - pval, **gaussian_kwargs)


def compare(imgs, ref_mask, image_comparison_type: str, gaussian_kwargs) -> (numpy.ndarray, numpy.ndarray):
    """
    compares images for "reference" and "treated" groups (pixelwise comparison),
    returns: p_values (false positive probability of test), and distance metric
    """
    # there are basically 2 methods to compare the images in batch:
    # by pairwise comparison or difference comparison
    if image_comparison_type == 'independent':
        _, pval = calculate_pixwise_t_ind(imgs, ref_mask)
        mdif = calculate_pixwise_difference_of_means(imgs, ref_mask)
    elif image_comparison_type == 'pairwise':
        _, pval = calculate_pixwise_t_rel(imgs, ref_mask)
        mdif = calculate_pixwise_mean_of_difference(imgs, ref_mask)
    else:
        raise ValueError("image_comparison_type must be one of {'independent', 'pairwise'}")
    pval = smooth_out_p_values(pval, gaussian_kwargs)
    return pval, mdif


def mask_gen(srfi: info_classes.segmentation_result_folder_info_like,
             metas: pandas.DataFrame) -> typing.Iterator[typing.Tuple[numpy.ndarray, dict]]:
    """yields pairs of (structure_mask, mask_metadata) loaded from anatomical atlas"""
    oi = srfi.ontology_folder_info().ontology_info(metas['frame'].iloc[0])
    rt = oi.tree().getroot()
    for node in rt.iter('structure'):
        fname = node.attrib['filename']
        mask = oi.open_mask_relative(fname)
        yield mask, node.attrib


def plot_pval_mdif(
        spec: dict,
        pval: numpy.ndarray,
        mdif: numpy.ndarray,
        metas: pandas.DataFrame,
        structure_mask: numpy.ndarray,
        structure_info: dict,
        plot_folder: pathlib.Path
        ) -> None:
    """
    plots and saves visualization of comparison p-values (pval) and distance metric (mdif) between groups,
    with anatomical mask (structure_mask) contour on top
    """
    fig, axs = plt.subplots(1, 2, **spec['figure_kwargs'])
    axs = axs.flatten()
    im0 = axs[0].imshow(pval, cmap='viridis_r')
    im1 = axs[1].imshow(mdif, cmap='coolwarm', norm=colors.TwoSlopeNorm(vcenter=0, vmin=None, vmax=None))
    plt.colorbar(im0, ax=axs[0], **spec['colorbar_kwargs'])
    plt.colorbar(im1, ax=axs[1], **spec['colorbar_kwargs'])
    im0.set_clim(0.0, 0.1)
    axs[0].contour(structure_mask, **spec['contour_kwargs'])
    axs[1].contour(structure_mask, **spec['contour_kwargs'])
    d = {**metas[~ metas['is_ref']].iloc[0], **structure_info}
    title = spec['plot_title'].format(**d)
    path = plot_folder / spec['plot_path'].format(**d)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title, **spec['plot_title_kwargs'])
    fig.savefig(path, **spec['savefig_kwargs'])
    plt.close(fig)


def mean_std_npx_over_bool_mask(imgs: numpy.ndarray, bool_mask: "numpy.ndarray[bool]") -> \
        typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """returns: mean, std, number_of_pixels_in_mask for each image"""
    npx = numpy.sum(bool_mask)
    if npx < 1:  # this is to suppress warnings, not to filter small masks.
        mean = std = numpy.tile(numpy.nan, len(imgs))
    else:
        pixels = numpy.array([im[bool_mask] for im in imgs])
        mean = numpy.mean(pixels, axis=-1)
        std = numpy.std(pixels, axis=-1)
    npx = numpy.tile(npx, len(imgs))
    return mean, std, npx


def stats_over_structure_mask(
        pval_thresholds: typing.Tuple[float],
        pval: numpy.ndarray,
        imgs: numpy.ndarray,
        metas: pandas.DataFrame,
        structure_mask: "numpy.ndarray[bool]",
        structure_info: dict
        ) -> pandas.DataFrame:
    """
    makes masks of significant values (where `pval > pval_thresholds[i]`)
    combines them with `structure_mask` (logical and), then computes mean, std, and mask size.
    returns: table_of_statistics: a combination of "metas" and computed stats.
    """
    stat = metas.copy()
    stat['structure'] = structure_info['name']
    for t in pval_thresholds:
        out = mean_std_npx_over_bool_mask(imgs, (pval < t) & structure_mask)
        stat[f'mean (p <{t})'] = out[0]
        stat[f'std (p <{t})'] = out[1]
        stat[f'px (p <{t})'] = out[2]
    return stat


def main(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like,
         batch_range: slice = None, save_intersection_images: bool = True):
    """for each batch of animals (see `pipelines.batch_for_comparison`)
    1. compare images with specified statistic (see results folder configuration, `comparison/image_comparison_type`)
    2. plot p-values (false positive rate), difference between images (if `save_intersection_images=True`)
    3. for differen p-value levels (see configuration `comparison/pval_thesholds`),
    calculate number of significant pixels, mean and standard deviation.
    calculated statistics are stored in "pickled" DataFrames,
    they need to be aggregated by `pipelines.compare_and_segment.collect_segmentation_results` function

    NB: one can specify a set of anatomical structures to use in the file
    at location given by `segmentation_results_folder_info.structure_list_path()`.
    the structures must be provided as newline-separated list.

    NB: one can also make "mask_permutation.py" script that will specify how to permute structure mask
    (2d numpy boolean array) the script must contain entry
    function `mask_permutation(x: numpy.ndarray) -> numpy.ndarray`.
    The location of such file is specified by `segmentation_results_folder_info.mask_permutation_path()`

    You can influence how the atlas mask is applied to the images by editing `mask_permutation.py`
    file in the results folder.
    """
    plt.ioff()
    srfi = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)
    plot_folder = get_plot_folder(srfi) if save_intersection_images else None
    srfi.segmentation_temp().mkdir(exist_ok=True)
    spec = srfi.configuration()['comparison']
    mask_permutation = load_mask_permutation(srfi)
    selected_structures = get_selected_structures(srfi.structure_list_path())
    bgen = batches_gen(srfi, batch_range)
    n_batches = next(bgen)

    progress_bar = tqdm.tqdm(leave=False, total=n_batches)
    for batch_no, ref_mask, batch in bgen:
        progress_bar.set_postfix_str(str(batch_no))
        progress_bar.update()

        imgs, metas = get_imgs_metas(srfi, batch)
        metas['is_ref'] = ref_mask[batch != 'nan']
        imgs = normalize_imgs(imgs, metas, spec)
        pval, mdif = compare(imgs, metas["is_ref"], spec["image_comparison_type"], spec["pvalue_smoothing_kwargs"])

        stats = []
        for structure_mask, structure_info in mask_gen(srfi, metas):
            if structure_info["name"] not in selected_structures:
                continue
            structure_mask = mask_permutation(structure_mask)
            if save_intersection_images:
                plot_pval_mdif(spec, pval, mdif, metas, structure_mask, structure_info, plot_folder)
            stats.append(
                stats_over_structure_mask(spec["pval_thresholds"], pval, imgs, metas, structure_mask, structure_info)
            )
        stats = pandas.concat(stats, axis=0)
        pickle_path = srfi.segmentation_temp()/f'{batch_no}.pickle'
        stats.to_pickle(str(pickle_path))

    progress_bar.close()
    plt.ion()


# ==============================================================================================
def collect_segmentation_results(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like,
                                 delete_temporary_folder=False):
    """
    Collects "pickled" results of `pipelines.compare_and_segment.main` function into a `segm_result.txt` table.
    Deletes columns, that are not specified in `comparison/take_columns_to_summary` entry of configuration.
    Based on `batching` entries of the configuration, deletes excessive rows, corresponding to "reference" images
    For "reference" images zeroes out values that correspond to
    significant thresholds from `comparison/pvalue_thresholds`
    """
    srfi = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)
    spec = srfi.configuration()
    temp_folder = srfi.segmentation_temp()
    t = join_pickles(temp_folder)
    t = t[spec["comparison"]["take_columns_to_summary"]]
    t = drop_duplicates(t, spec["batching"])
    t = zero_self_comparison_stats(t, spec["comparison"]["pvalue_thresholds"])
    save_path = srfi.table_folder() / 'segm_result.txt'
    t.to_csv(save_path, sep='\t')
    if delete_temporary_folder:
        miscellaneous_utils.delete_folder(temp_folder)

    
def join_pickles(folder: pathlib.Path) -> pandas.DataFrame:
    """loads and concatenates pickled file from single folder. does not check path validity"""
    chunks = []
    for path in folder.iterdir():
        chunks.append(pandas.read_pickle(path))
    t = pandas.concat(chunks, axis=0, ignore_index=True, copy=False)
    return t


def drop_duplicates(t: pandas.DataFrame, bspec: dict) -> pandas.DataFrame:
    """
    (in-place) drops duplicate rows from DataFrame, according to "batching"
    arguments from segmentation folder configuration:
    columns that correspond to "batching" options are set as DataFrame index,
    then duplicate index entries are removed.
    these duplicate entries come from multiple comparisons with single group
    """
    index_col = ['structure'] + bspec['compare_by'] + bspec['match_by'] + bspec['batch_by']
    t.set_index(index_col, inplace=True)
    dupl = t.index.duplicated(keep='first')  # bool mask
    dupl = numpy.arange(len(dupl))[dupl]  # int ordinals
    t.reset_index(inplace=True)
    t.drop(index=dupl, inplace=True)
    t.set_index(index_col, inplace=True)
    t.sort_index(axis=0, inplace=True)
    return t


def zero_self_comparison_stats(table: pandas.DataFrame,
                               pvalue_thresholds: typing.Tuple[float]) -> pandas.DataFrame:
    """
    "reference" images statistics for significant p-value thresholds (<0.99)
    are set to zero/nan
    """
    ref_idx = table[table['is_ref']].index
    for t in pvalue_thresholds:
        if t < 0.99:
            clmn1 = [f"mean (p <{t})", f"std (p <{t})", ]
            clmn2 = [f"px (p <{t})", ]
            table.loc[ref_idx, clmn1] = numpy.nan
            table.loc[ref_idx, clmn2] = 0
    return table
