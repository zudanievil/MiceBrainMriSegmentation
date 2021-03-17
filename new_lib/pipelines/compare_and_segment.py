import sys
import datetime
import tqdm
import numpy
import pandas
import scipy.stats
import skimage.filters
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ..core import info_classes


PLOT_FOLDER_MAX_LENGTH = 100


def load_mask_permutation(srfi: info_classes.segmentation_result_folder_info_like) -> callable:
    p = srfi.mask_permutation_path()
    if p.exists():
        with p.open('rt') as f:
            exec(f.read())
        return locals()['mask_permutation']
    else:
        return lambda x, *y: x


def batches_gen(srfi: info_classes.segmentation_result_folder_info_like, batch_range: slice = None):
    path = srfi.batches_path()
    batches = pandas.read_csv(path, sep='\t', index_col=0, header=[0, 1], na_values='NA',
                              true_values='True', false_values='False')
    ref_mask = batches.columns.get_level_values(0).to_numpy()
    ref_mask = ref_mask == 'True'
    batches = batches.iloc[batch_range] if batch_range else batches
    yield len(batches)
    for i in batches.index:
        yield i, ref_mask, batches.iloc[i].to_numpy()


def get_imgs_metas(sfri: info_classes.segmentation_result_folder_info_like, batch, ref_mask, spec):
    ifi = sfri.image_folder_info()
    metas = []
    imgs = []
    for name in batch.tolist():
        ii = ifi.image_info(name)
        metas.append(ii.metadata())
        imgs.append(ii.cropped_image())
    metas = pandas.DataFrame(metas)
    metas['is_ref'] = ref_mask
    imgs = numpy.stack(imgs, axis=0)
    ref_key = spec['normalize_image_by']
    imgs /= metas[ref_key].to_numpy()[..., numpy.newaxis, numpy.newaxis]
    return imgs, metas


def calculate_pixwise_difference_of_means(imgs, metas) -> numpy.ndarray:
    is_ref = metas['is_ref']
    mdif = imgs[~ is_ref].mean(axis=0) - imgs[is_ref].mean(axis=0)
    return mdif


def calculate_pixwise_mean_of_difference(imgs, metas) -> numpy.ndarray:
    is_ref = metas['is_ref']
    mdif = (imgs[~ is_ref] - imgs[is_ref]).mean(axis=0)
    return mdif


def calculate_pixwise_pvalue_t_ind(imgs, metas) -> numpy.ndarray:
    is_ref = metas['is_ref']
    tval, pval = scipy.stats.ttest_ind(imgs[~ is_ref], imgs[is_ref], axis=0, equal_var=False)
    pval = 1 - skimage.filters.gaussian(1 - pval, sigma=1, truncate=1)
    return pval


def calculate_pixwise_pvalue_t_rel(imgs, metas) -> numpy.ndarray:
    is_ref = metas['is_ref']
    zval, pval = scipy.stats.ttest_rel(imgs[~ is_ref], imgs[is_ref], axis=0)
    pval = 1 - skimage.filters.gaussian(1 - pval, sigma=1, truncate=1)
    return pval


def compare(imgs, metas, spec):
    # there are basically 2 methods to compare the images in batch:
    # by pairwise comparison or difference comparison
    if spec['image_comparison_type'] == 'independent':
        pval = calculate_pixwise_pvalue_t_ind(imgs, metas)
        mdif = calculate_pixwise_difference_of_means(imgs, metas)
    elif spec['image_comparison_type'] == 'pairwise':
        pval = calculate_pixwise_pvalue_t_rel(imgs, metas)
        mdif = calculate_pixwise_mean_of_difference(imgs, metas)
    else:
        raise ValueError("image_comparison_type must be one of {'independent', 'pairwise'}")
    return pval, mdif


def get_mask_gen(srfi: info_classes.segmentation_result_folder_info_like,
                 metas: dict, spec: dict) -> info_classes.ontology_info_iterator_type:
    oi = srfi.ontology_folder_info().ontology_info(metas['frame'].iloc[0])
    rt = oi.tree().getroot()
    for node in rt.iter('structure'):
        fname = node.attrib['filename']
        mask = oi.open_mask(fname)
        yield mask, node.attrib


def plot_pval_mdif(spec, pval, mdif, metas, structure_mask, structure_info, plot_folder):
    fig, axs = plt.subplots(1, 2, **spec['figure_kwargs'])
    axs = axs.flatten()
    im0 = axs[0].imshow(pval, cmap='viridis_r')
    im1 = axs[1].imshow(mdif, cmap='coolwarm', norm=colors.TwoSlopeNorm(vcenter=0, vmin=None, vmax=None))
    # https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html
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


def mean_std_npx_over_bool_mask(imgs, bool_mask) -> 'tuple[numpy.ndarray]':
    npx = numpy.sum(bool_mask)
    l = len(imgs)
    if npx < 2:
        mean = std = numpy.tile(numpy.nan, l)
    else:
        bool_mask = bool_mask[numpy.newaxis, ...]
        sum = numpy.sum(bool_mask * imgs, axis=(1, 2))
        mean = sum / npx
        i = bool_mask * (imgs - mean[..., numpy.newaxis, numpy.newaxis]) ** 2  # i for intermediate
        std = numpy.sqrt(numpy.sum(i, axis=(1, 2)) / (npx - 1))
    npx = numpy.tile(npx, l)
    return mean, std, npx


def stats_over_structure_mask(spec, pval, imgs, metas, structure_mask, structure_info) -> pandas.DataFrame:
    stat = metas.copy()
    not_ref = ~ metas['is_ref']
    stat['group'] = metas['group'][not_ref].iloc[0]
    stat['structure'] = structure_info['name']
    for t in spec['pval_thresholds']:
        out = mean_std_npx_over_bool_mask(imgs, (pval < t) * structure_mask)
        stat[f'mean (p <{t})'] = out[0]
        stat[f'std (p <{t})'] = out[1]
        stat[f'px (p <{t})'] = out[2]
    return stat


def main(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like,
         batch_range: slice = None, save_intersection_images: bool = True):
    srfi = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)

    if save_intersection_images:
        plot_folder = srfi.plot_folder()
        assert len(str(plot_folder)) <= PLOT_FOLDER_MAX_LENGTH,\
            f'plot folder path length <= {PLOT_FOLDER_MAX_LENGTH} is recommended'
        plot_folder.mkdir(exist_ok=True)
    srfi.segmentation_temp().mkdir(exist_ok=True)
    spec = srfi.specification()['comparison']
    mask_permutation = load_mask_permutation(srfi)
    bgen = batches_gen(srfi, batch_range)
    n_batches = bgen.__next__()
    progress_bar = tqdm.tqdm(leave=False, total=n_batches, file=sys.stdout)
    for batch_no, ref_mask, batch in bgen:
        progress_bar.update()
        progress_bar.set_postfix_str(f'\n{batch_no} | {datetime.datetime.now()} | {batch}\n')
        imgs, metas = get_imgs_metas(srfi, batch, ref_mask, spec)
        pval, mdif = compare(imgs, metas, spec)
        stats = []
        mask_gen = get_mask_gen(srfi, metas, spec)
        for structure_mask, structure_info in mask_gen:
            structure_mask = mask_permutation(structure_mask)
            if save_intersection_images:
                plot_pval_mdif(spec, pval, mdif, metas, structure_mask, structure_info, plot_folder)
            stats.append(stats_over_structure_mask(spec, pval, imgs, metas, structure_mask, structure_info))
        stats = pandas.concat(stats, axis=0)
        pickle_path = srfi.segmentation_temp()/(batch_no + '.pickle')
        stats.to_pickle(pickle_path)
