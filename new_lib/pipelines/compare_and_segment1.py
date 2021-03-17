from ..core import info_classes
# def assert_path_is_short(path: pathlib.Path) -> None:
#     if sys.platform.startswith('win') and len(str(path)) > 100:
#         raise AssertionError('On windows long file paths (> 260 chars) are not allowed.'
#                              'Since masks are nested hierarchically in directories,'
#                              'it is best if {project}/segm path is < 100 chars')


# def assert_no_old_pickles(segmentation_temp: pathlib.Path, batch_range: slice = None) -> None:
#     # for i in batches.index.to_list():
#     #     if (folder / f'{i}.pickle').exists():
#     #         raise AssertionError(f'{folder} should not contain pickles for specified batches')
#     pass





# def read_structure_set(project: pathlib.Path, spec_name: str) -> 'set[str]':
#     p = project / 'specs' / spec_name / 'segm' / 'structures.txt'
#     structure_set = None
#     if p.exists():
#         with p.open('rt') as f:
#             structure_set = {s.strip() for s in f.readlines()}
#         if '' in structure_set:
#             structure_set.remove('')
#     return structure_set


# def validate_structure_set(structure_set: set, masks_folder: pathlib.Path,
#                            mismatch_policy: "{'strict', 'ignore'}" = 'strict'):
#     if not structure_set:
#         return None
#     if mismatch_policy not in {'strict', 'ignore'}:
#         raise NotImplementedError(f'mismatch policy "{mismatch_policy}" not implemented')
#     ont = ontology.Ontology(masks_folder, 'placeholder')
#     root = ont.default_xml_tree.getroot()
#     default_structure_set = {s.attrib['name'] for s in root.iter('structure')}
#     del root
#     for structure in structure_set:
#         if structure not in default_structure_set:
#             if mismatch_policy == 'strict':
#                 p = ont.folder / 'default.xml'
#                 raise AssertionError(f'"{structure}" does not match any default\n'
#                                      f'tree structure of ontology from {p}')


# def load_imgs_metas(project, batch, ref_mask) -> (np.ndarray, np.ndarray):
#     imgs = load_image_batch(project, batch)
#     metas = load_meta_batch(project, batch)
#     nan_mask = batch == 'nan'
#     metas['is_ref'] = ref_mask[~ nan_mask]
#     imgs = imgs / metas['reference'].to_numpy()[..., np.newaxis, np.newaxis]  # batch dimension is the first
#     return imgs, metas


# def calc_pixwise_difference_of_means(imgs, metas) -> np.ndarray:
#     is_ref = metas['is_ref']
#     mdif = imgs[~ is_ref].mean(axis=0) - imgs[is_ref].mean(axis=0)
#     return mdif


# def calc_pixwise_mean_of_difference(imgs, metas) -> np.ndarray:
#     is_ref = metas['is_ref']
#     mdif = (imgs[~ is_ref] - imgs[is_ref]).mean(axis=0)
#     return mdif


# def calculate_pixwise_pvalue_t_ind(imgs, metas) -> np.ndarray:
#     is_ref = metas['is_ref']
#     # imgs = skimage.filters.gaussian(imgs, **_LOC['segm.gauss_filt_kw'])
#     tval, pval = scipy.stats.ttest_ind(imgs[~ is_ref], imgs[is_ref], **_LOC['segm.ttest_kw'])
#     pval = 1 - skimage.filters.gaussian(1 - pval, sigma=1, truncate=1)
#     return pval


# def calculate_pixwise_pvalue_t_rel(imgs, metas) -> np.ndarray:
#     is_ref = metas['is_ref']
#     imgs = skimage.filters.gaussian(imgs, **_LOC['segm.gauss_filt_kw'])
#     zval, pval = scipy.stats.ttest_rel(imgs[~ is_ref], imgs[is_ref], axis=0)
#     return pval


# def get_search_mask(shape: tuple) -> 'np.ndarray[bool]':  # TODO: implement control with _LOC
#     mask = np.zeros(shape, dtype=bool)
#     mask[:, shape[1] // 2:] = True
#     return mask


# def fetch_ontology(metas, masks_folder) -> ontology.Ontology:
#     not_ref = ~ metas['is_ref']
#     frame = metas['frame'][not_ref].iloc[0]
#     folder = masks_folder
#     ont = ontology.Ontology(folder, frame)
#     return ont


# def plot_pval_mdif(plot_folder, structure_mask, structure_info,  metas, pval, mdif) -> File:
#     fig, axs = plt.subplots(1, 2, **_LOC['segm.subplots_kw'])
#     axs = axs.flatten()
#     im0 = axs[0].imshow(pval, cmap='viridis_r')
#     im1 = axs[1].imshow(mdif, cmap='coolwarm', norm=colors.TwoSlopeNorm(vcenter=0, vmin=None, vmax=None))
#     # https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html
#     plt.colorbar(im0, ax=axs[0], **_LOC['segm.colorbar_kw'])
#     plt.colorbar(im1, ax=axs[1], **_LOC['segm.colorbar_kw'])
#     im0.set_clim(0.0, 0.1)
#     axs[0].contour(structure_mask, **_LOC['segm.contour_kw'])
#     axs[1].contour(structure_mask, **_LOC['segm.contour_kw'])
#     d = {**metas.iloc[0].to_dict(), **structure_info}
#     title = _LOC['segm.plot_title'].format(**d)
#     path = pathlib.Path(plot_folder / _LOC['segm.plot_path'].format(**d))
#     path.parent.mkdir(parents=True, exist_ok=True)
#     fig.suptitle(title, **_LOC['segm.plot_title_kw'])
#     fig.savefig(path, **_LOC['segm.savefig_kw'])
#     plt.close(fig)


# def mean_std_npx_over_bool_mask(imgs, bool_mask) -> 'tuple[np.ndarray]':
#     npx = np.sum(bool_mask)
#     l = len(imgs)
#     if npx < 2:
#         mean = std = np.tile(np.nan, l)
#     else:
#         bool_mask = bool_mask[np.newaxis, ...]
#         sum = np.sum(bool_mask * imgs, axis=(1, 2))
#         mean = sum / npx
#         i = bool_mask * (imgs - mean[..., np.newaxis, np.newaxis]) ** 2  # i for intermediate
#         std = np.sqrt(np.sum(i, axis=(1, 2)) / (npx - 1))
#     npx = np.tile(npx, l)
#     return mean, std, npx


# def make_stat(structure_mask, structure_info, metas, imgs, pval) -> pd.DataFrame:
#     # takes from the outer scope: imgs, pval, structure_mask
#     cls = segment_batches
#     stat = metas.copy()
#     not_ref = ~ metas['is_ref']
#     stat['group'] = metas['group'][not_ref].iloc[0]
#     stat['structure'] = structure_info['name']
#     for t in _LOC['pval_thresholds']:
#         out = mean_std_npx_over_bool_mask(imgs, (pval < t) * structure_mask)
#         stat[f'mean (p <{t})'] = out[0]
#         stat[f'std (p <{t})'] = out[1]
#         stat[f'px (p <{t})'] = out[2]
#     return stat


def get_project_info(project: 'info_classes.ProjectInfo or pathlib.Path or str') -> info_classes.ProjectInfo:
    if isinstance(project, info_classes.ProjectInfo):
        project_info = project
    elif isinstance(project, (pathlib.Path, str)):
        project = pathlib.Path(project)
        project_info = info_classes.ProjectInfo.read(project)
    else:
        raise TypeError(f'{project} is of invalid type')
    return project_info


def get_segmentation_args(project_info: info_classes.ProjectInfo) -> argparse.Namespace:
    path = project_info.spec_folder / 'segmentation_args.yml'
    with path.open('rt'):
        args = yaml.safe_load(path)
    # augment with project_info fields
    return args


def get_structure_set(project_info) -> 'set or UniversumSet':
    try:
        sturcture_set = read_structure_set()
        validate_structure_set()
    except FileNotFoundError:
        structure_set = UniversumSet()
    return structure_set


def read_structure_set():
    raise NotImplementedError


def validate_structure_set():
    raise NotImplementedError



def get_batches(project_info, batch_range):
    path = project_info...
    batches, ref_mask = read_batches()
    batches = batches[batch_range] if batch_range else batches
    return batch_range, ref_mask

def read_batches(path: pathlib.Path) -> (np.ndarray, np.ndarray):
    batches = pd.read_csv(path, sep='\t', na_values='NA', header=[0, 1])
    ref_mask = np.array(batches.columns.get_level_values(0)) == 'True'
    batches = batches.to_numpy(dtype=str)
    return batches, ref_mask


def get_imgs_metas(project_info, batch, ref_mask, args):
    paths = ...
    imgs = read_imgs_batch(paths)
    metas = read_metas_batch(paths)
    imgs *= (metas['reference'] / metas['ER_head_median']).to_numpy()[..., np.newaxis, np.newaxis]
    return imgs, metas

def compare(imgs, metas, args):
    # there are basically 2 methods to compare the images in batch:
    # by pairwise comparison or difference comparison
    if args.image_comparison_type == 'means':
        pval = calculate_pixwise_pvalue_t_ind(imgs, metas)
        mdif = calc_pixwise_difference_of_means(imgs, metas)
    elif args.image_comparison_type == 'pairwise':
        pval = calculate_pixwise_pvalue_t_rel(imgs, metas)
        mdif = calc_pixwise_mean_of_difference(imgs, metas)

def get_masks_generator(ontology_info, structure_set, args):
    pass

def masks_generator():
    pass

def make_pval_mdif_plot():
    pass

def stat_over_structure_mask():
    pass

def save_stats():
    save_path = args.temp_folder / f'{i}.pickle'
    stats.to_pickle(save_path, compression=None)


def main(project: 'info_classes.ProjectInfo or pathlib.Path or str',
         mask_permutation: 'callable' = lambda x: x,
         batch_range: slice = None,
         save_intersection_images: bool = True) -> File:
    project_info = get_project_info(project)
    structure_set = get_structure_set(project_info)
    args = get_segmentation_args(project_info)

    if save_intersection_images:
        assert_path_is_short(args.plot_folder)
        args.plot_folder.mkdir(exist_ok=True)
    args.stats_folder.mkdir(exist_ok=True)

    for batch_no, batch, ref_mask in batches_generator(project_info, batch_range):
        imgs, metas = get_imgs_metas(args, batch, ref_mask)
        pval, mdif = compare(args, imgs, metas)
        mask_gen = get_masks_generator(args, structure_set)
        stats = []
        for structure_mask, structure_info in mask_gen:
            structure_mask = mask_permutation(structure_mask)
            a = args, pval, mdif, metas, structure_mask, structure_info 
            if save_intersection_images:
                make_pval_mdif_plot(*a)
            stats.append(stat_over_structure_mask(*a))
        stats = pd.concat(stats, axis=0)
        save_stats(batch_no, stats)
