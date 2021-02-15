import datetime
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy.stats
import skimage.filters
import yaml
from matplotlib import colors, pyplot as plt

from lib import pattern_utils, ontology

_LOC = dict()
_GLOB = dict()
_IMPORTS_THAT_NEED_CONFIG = (pattern_utils, ontology)

File = type(None)  # type alias. means function writes result to files


class find_batches_for_segmentation:
    """
    static class that represents calculation.
    call() method is the entry point, see docs of call() method for details.
    """
    @staticmethod
    def table_of_images(project: pathlib.Path) -> pd.DataFrame:
        regex = pattern_utils.fstring_to_regex(_GLOB['file_naming_convention'])
        table = []
        columns = ('name', *regex.groupindex)
        for image_path in (project / 'img').iterdir():
            image_name = image_path.stem
            table.append((image_name, *regex.match(image_name).groups()))
        table = pd.DataFrame(table, columns=columns)
        return table

    @staticmethod
    def split_to_groups(table: pd.DataFrame) -> pd.DataFrame:  # TODO: this needs config
        new_index = _LOC['compare_by'] + _LOC['match_by'] + _LOC['batch_by']
        reference_value = str(_LOC['reference_value'])
        table = table.set_index(new_index)[['name', ]].unstack(level=-1)
        table.sort_index(axis=0, inplace=True)
        table.columns.rename('is_reference', level=0, inplace=True)
        ref_table = table.loc[reference_value].rename(columns={'name': True})
        table = table.drop(index=[reference_value])
        table.rename(columns={'name': False}, inplace=True)
        idx = table.index.get_level_values(0).unique()
        chunks = []
        for i in idx:
            chunks.append(pd.concat([table.loc[i], ref_table], axis=1))
        table = pd.concat(chunks, axis=0)
        return table

    @staticmethod
    def call(project: pathlib.Path, spec_name: str = 'default') -> File:
        """
        makes table where each row is filenames, compared against each other.
        The columns' headers indicate whether image is from reference group.
        Writes table into {project}/specs/{spec_name}/segm/batches.txt.
        """
        cls = find_batches_for_segmentation
        t = cls.table_of_images(project)
        t = cls.split_to_groups(t)
        t.reset_index(drop=True, inplace=True)
        save_path = project / 'specs' / spec_name / 'segm' / 'batches.txt'
        t.to_csv(save_path, sep='\t', na_rep='NA', index=False)


def load_image_batch(project, batch: np.ndarray) -> np.ndarray:
    imgs = []
    for name in batch:
        if name == 'nan':
            continue
        path = project / 'img_cropped' / (name + '.npy')
        imgs.append(np.load(path, fix_imports=False))
    imgs = np.stack(imgs, axis=0)
    return imgs


def load_meta_batch(project, batch: np.ndarray) -> pd.DataFrame:
    metas = []
    for name in batch:
        if name == 'nan':
            continue
        with (project / 'meta' / (name + '.yml')).open('rt') as f:
            metas.append(yaml.safe_load(f.read()))
    metas = pd.DataFrame(metas)
    return metas


class segment_batches:
    """
    static class that represents calculation.
    call() method is the entry point, see docs of call() method for details.
    """

    @staticmethod
    def assert_path_is_short(path: pathlib.Path) -> None:
        if sys.platform.startswith('win') and len(str(path)) > 100:
            raise AssertionError('On windows long file paths (> 260 chars) are not allowed.'
                                 'Since masks are nested hierarchically in directories,'
                                 'it is best if {project}/segm path is < 100 chars')

    @staticmethod
    def assert_no_old_pickles(segmentation_temp: pathlib.Path, batch_range: slice = None) -> None:
        # for i in batches.index.to_list():
        #     if (folder / f'{i}.pickle').exists():
        #         raise AssertionError(f'{folder} should not contain pickles for specified batches')
        pass

    @staticmethod
    def read_batches(project: pathlib.Path, spec_name: str, batch_range: slice = None) -> (np.ndarray, np.ndarray):
        batches = pd.read_csv(project / 'specs' / spec_name / 'segm' / 'batches.txt',
                              sep='\t', na_values='NA', header=[0, 1])
        if batch_range:
            batches = batches.iloc[batch_range]
        ref_mask = np.array(batches.columns.get_level_values(0)) == 'True'
        batches = batches.to_numpy(dtype=str)
        return batches, ref_mask

    @staticmethod
    def read_structure_set(project: pathlib.Path, spec_name: str) -> 'set[str]':
        p = project / 'specs' / spec_name / 'segm' / 'structures.txt'
        structure_set = None
        if p.exists():
            with p.open('rt') as f:
                structure_set = {s.strip() for s in f.readlines()}
        if '' in structure_set:
            structure_set.remove('')
        return structure_set

    @staticmethod
    def validate_structure_set(structure_set: set, masks_folder: pathlib.Path,
                               mismatch_policy: "{'strict', 'ignore'}" = 'strict'):
        if mismatch_policy not in {'strict', 'ignore'}:
            raise NotImplementedError(f'mismatch policy "{mismatch_policy}" not implemented')
        ont = ontology.Ontology(masks_folder, 'placeholder')
        root = ont.default_xml_tree.getroot()
        default_structure_set = {s.attrib['name'] for s in root.iter('structure')}
        del root
        for structure in structure_set:
            if structure not in default_structure_set:
                if mismatch_policy == 'strict':
                    p = ont.folder / 'default.xml'
                    raise AssertionError(f'"{structure}" does not match any default\n'
                                         f'tree structure of ontology from {p}')

    @staticmethod
    def load_imgs_metas(project, batch, ref_mask) -> (np.ndarray, np.ndarray):
        imgs = load_image_batch(project, batch)
        metas = load_meta_batch(project, batch)
        nan_mask = batch == 'nan'
        metas['is_ref'] = ref_mask[~ nan_mask]
        imgs = imgs / metas['reference'].to_numpy()[..., np.newaxis, np.newaxis]  # batch dimension is the first
        return imgs, metas

    @staticmethod
    def calc_pixwise_difference_of_means(imgs, metas) -> np.ndarray:
        is_ref = metas['is_ref']
        mdif = imgs[~ is_ref].mean(axis=0) - imgs[is_ref].mean(axis=0)
        return mdif

    @staticmethod
    def calc_pixwise_mean_of_difference(imgs, metas) -> np.ndarray:
        is_ref = metas['is_ref']
        mdif = (imgs[~ is_ref] - imgs[is_ref]).mean(axis=0)
        return mdif

    @staticmethod
    def calculate_pixwise_pvalue(imgs, metas) -> np.ndarray:
        is_ref = metas['is_ref']
        imgs = skimage.filters.gaussian(imgs, **_LOC['segm.gauss_filt_kw'])
        tval, pval = scipy.stats.ttest_ind(imgs[~ is_ref], imgs[is_ref], **_LOC['segm.ttest_kw'])
        return pval

    @staticmethod
    def get_search_mask(shape: tuple) -> 'np.ndarray[bool]':  # TODO: implement control with _LOC
        mask = np.zeros(shape, dtype=bool)
        mask[:, shape[1] // 2:] = True
        return mask

    @staticmethod
    def fetch_ontology(metas, masks_folder) -> ontology.Ontology:
        not_ref = ~ metas['is_ref']
        frame = metas['frame'][not_ref].iloc[0]
        folder = masks_folder
        ont = ontology.Ontology(folder, frame)
        return ont

    @staticmethod
    def plot_pval_mdif(plot_folder, structure_mask, structure_info,  metas, pval, mdif) -> File:
        fig, axs = plt.subplots(1, 2, **_LOC['segm.subplots_kw'])
        axs = axs.flatten()
        im0 = axs[0].imshow(pval, cmap='viridis_r')
        im1 = axs[1].imshow(mdif, cmap='coolwarm', norm=colors.TwoSlopeNorm(vcenter=0, vmin=None, vmax=None))
        # https://matplotlib.org/3.1.1/tutorials/colors/colormapnorms.html
        plt.colorbar(im0, ax=axs[0], **_LOC['segm.colorbar_kw'])
        plt.colorbar(im1, ax=axs[1], **_LOC['segm.colorbar_kw'])
        im0.set_clim(0.0, 0.1)
        axs[0].contour(structure_mask, **_LOC['segm.contour_kw'])
        axs[1].contour(structure_mask, **_LOC['segm.contour_kw'])
        d = {**metas.iloc[0].to_dict(), **structure_info}
        title = _LOC['segm.plot_title'].format(**d)
        path = pathlib.Path(plot_folder / _LOC['segm.plot_path'].format(**d))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.suptitle(title, **_LOC['segm.plot_title_kw'])
        fig.savefig(path, **_LOC['segm.savefig_kw'])
        plt.close(fig)

    @staticmethod
    def mean_std_npx_over_bool_mask(imgs, bool_mask) -> 'tuple[np.ndarray]':
        npx = np.sum(bool_mask)
        l = len(imgs)
        if npx < 2:
            mean = std = np.tile(np.nan, l)
        else:
            bool_mask = bool_mask[np.newaxis, ...]
            sum = np.sum(bool_mask * imgs, axis=(1, 2))
            mean = sum / npx
            i = bool_mask * (imgs - mean[..., np.newaxis, np.newaxis]) ** 2  # i for intermediate
            std = np.sqrt(np.sum(i, axis=(1, 2)) / (npx - 1))
        npx = np.tile(npx, l)
        return mean, std, npx

    @staticmethod
    def make_stat(structure_mask, structure_info, metas, imgs, pval) -> pd.DataFrame:
        # takes from the outer scope: imgs, pval, structure_mask
        cls = segment_batches
        stat = metas.copy()
        not_ref = ~ metas['is_ref']
        stat['group'] = metas['group'][not_ref].iloc[0]
        stat['structure'] = structure_info['name']
        for t in _LOC['pval_thresholds']:
            out = cls.mean_std_npx_over_bool_mask(imgs, (pval < t) * structure_mask)
            stat[f'mean (p <{t})'] = out[0]
            stat[f'std (p <{t})'] = out[1]
            stat[f'px (p <{t})'] = out[2]
        return stat

    @staticmethod
    def call(project: pathlib.Path, masks_folder: pathlib.Path,
             mask_permutation: 'callable' = lambda x: x,
             batch_range: slice = None,
             save_intersection_images: bool = True,
             spec_name: str = 'default') -> File:
        """
        Reads batches from {project}/specs/{spec_name}/segm/batches.txt
        Reads structure list from {project}/specs/{spec_name}/segm/structure.txt
        if the structure list is not found, does segmentation for all structures.
        :param save_intersection_images: saves renders of image comparison results
        with brain structure mask overlayed to the {project}/results/{spec_name}/segm
        :param mask_permutation: callable that is applied to structure mask (2d bool numpy array),
        the moment mask has been loaded.
        :param batch_range: if you want to use specific batches only (to resume work, for eg).
        :param spec_name: name of the specification in {project}/specs/segm/{spec_name}
        """
        cls = segment_batches
        segmentation_temp = project / f'.segmentation_temp_{spec_name}'
        segmentation_temp.mkdir(exist_ok=True)
        cls.assert_no_old_pickles(segmentation_temp, batch_range)  # TODO: delete this or not?
        if save_intersection_images:
            intersection_image_folder = project / 'results' / spec_name / 'segm'
            cls.assert_path_is_short(intersection_image_folder)
        structure_set = cls.read_structure_set(project, spec_name)
        cls.validate_structure_set(structure_set, masks_folder)
        batches, ref_mask = cls.read_batches(project, spec_name, batch_range)
        for i, batch in enumerate(batches):
            print(datetime.datetime.now(), 'starting: ', list(batch))
            imgs, metas = cls.load_imgs_metas(project, batch, ref_mask)
            pval = cls.calculate_pixwise_pvalue(imgs, metas)
            # mdif = cls.calc_pixwise_mean_of_difference(imgs, metas)
            mdif = cls.calc_pixwise_difference_of_means(imgs, metas)
            ont = cls.fetch_ontology(metas, masks_folder)
            stats = []
            for structure_mask, structure_info in ontology.masks_generator(ont):
                if structure_set:
                    if structure_info['name'] not in structure_set:
                        continue
                structure_mask = mask_permutation(structure_mask)
                print(datetime.datetime.now(), structure_info)
                if save_intersection_images:
                    cls.plot_pval_mdif(intersection_image_folder, structure_mask, structure_info, metas, pval, mdif)
                stats.append(cls.make_stat(structure_mask, structure_info, metas, imgs, pval))
            stats = pd.concat(stats, axis=0)
            save_path = segmentation_temp / f'{i}.pickle'
            stats.to_pickle(save_path, compression=None)
        print(datetime.datetime.now(), 'segmentation finished')


def print_structure_list(project: pathlib.Path, masks_folder: pathlib.Path, spec_name: str = 'default'):
    """
    prints the list of all the structures into the {project}/spec/{spec_name}/segm/structures.txt
    indents them with space to show the hierarchy
    :raises AssertionError: if structures.txt already exists
    """
    save_path = project / 'specs' / spec_name / 'segm' / 'structures.txt'
    if save_path.exists():
        raise AssertionError(f'{save_path} already exists')
    save_path.parent.mkdir(exist_ok=True)
    ont = ontology.Ontology(masks_folder, 'placeholder')
    root = ont.default_xml_tree.getroot()
    with save_path.open('wt') as f:
        for st in root.iter('structure'):
            s = ' ' * int(st.attrib['level']) + st.attrib['name']
            print(s, end='\n', file=f)


class collect_segmentation_results:
    """
    static class that represents calculation.
    call() method is the entry point, see docs of call() method for details.
    """

    @staticmethod
    def join_pickles(project: pathlib.Path, spec_name: str) -> pd.DataFrame:
        chunks = []
        for file in (project / f'.segmentation_temp_{spec_name}').iterdir():
            if file.suffix == '.pickle':
                chunks.append(pd.read_pickle(file))
        df = pd.concat(chunks, axis=0, ignore_index=True, copy=False)
        return df

    @staticmethod
    def drop_duplicates(t: pd.DataFrame) -> pd.DataFrame:
        index_col = ['structure'] + _LOC['compare_by'] \
                    + _LOC['match_by'] + _LOC['batch_by']
        t.set_index(index_col, inplace=True)
        dupl = t.index.duplicated(keep='first')  # bool mask
        dupl = np.arange(len(dupl))[dupl]  # int ordinals
        t.reset_index(inplace=True)
        t.drop(index=dupl, inplace=True)
        t.set_index(index_col, inplace=True)
        t.sort_index(axis=0, inplace=True)
        # we get ref_hour records per each hour
        # it is logical to make 1 ref_hour record per all hours
        # so we make a Multiindex table and delete all the rows with duplicate
        # index of (structure, hour, frame, animal)
        return t

    @staticmethod
    def zero_self_comparison_stats(t: pd.DataFrame) -> pd.DataFrame:
        idx = t[t['is_ref']].index
        clmn1 = ['mean (p <0.05)', 'std (p <0.05)', 'mean (p <0.01)', 'std (p <0.01)', ]
        clmn2 = ['px (p <0.05)', 'px (p <0.01)', ]
        t.loc[idx, clmn1] = np.nan
        t.loc[idx, clmn2] = 0
        # reference values cannot differ from themselves, so we set them to 0/nan
        return t

    @staticmethod
    def call(project: pathlib.Path, spec_name: str = 'default') -> File:
        cls = collect_segmentation_results
        t = cls.join_pickles(project, spec_name)
        t.drop(columns=_LOC['drop_columns_from_summary'], inplace=True)
        t = cls.drop_duplicates(t)
        t = cls.zero_self_comparison_stats(t)
        t.to_csv(project / 'results' / spec_name / 'segm_result.txt', sep='\t')


def plot_segmentation_results(project: pathlib.Path, spec_name: str,
                              save_plots_with_segmentation_images: bool = True) -> File:
    result_folder = project / 'results' / spec_name / 'segm'
    load_path = project / 'results' / spec_name / 'segm_result_refactored.txt'
    table = pd.read_csv(load_path, sep='\t',
                        index_col=['structure', _LOC['compare_by'][0]])
    structures = np.unique(table.index.get_level_values('structure'))
    for structure in structures:
        save_name = structure.replace('/', '_')
        if save_plots_with_segmentation_images:
            save_path = pattern_utils.find_file(save_name, result_folder).parent
        else:
            save_path = result_folder
        save_path /= (save_name + ' plot.png')

        data = table.loc[structure]
        fig, axs = plt.subplots(1, 1, figsize=_LOC['summary.figsize'])
        for i, t in enumerate(_LOC['pval_thresholds']):
            axs.errorbar(data.index - 0.3 * i, data[f'mean (p <{t})'], yerr=data[f'std (p <{t})'],
                         **_LOC['summary.plot_kw'][i])
        axs.set_xticks(data.index)
        axs.grid()
        axs.legend(loc='upper right')
        fig.savefig(save_path, **_LOC['summary.savefig_kw'])
        plt.close(fig)
        print(datetime.datetime.now(), save_path)


def refactor_summary(project: pathlib.Path, spec_name) -> File:
    """
    Use "compare_by" and pval thresholds to create a table with
    anatomical_structure/(mean_intensity, std, pvalue) axes
    ('segm_result.txt' -> 'segm_result_refactored.txt').
    The produced table is much more human-readable.
    """
    result_folder = project / 'results' / spec_name
    load_path = result_folder / 'segm_result.txt'
    save_path = result_folder / 'segm_result_refactored.txt'

    table = pd.read_csv(load_path, sep='\t')
    compare_by = _LOC['compare_by'][0]
    plot_cols = table.columns[[c.startswith(('mean', 'std', 'px')) for c in table.columns]].to_list()
    plot_cols += ['structure', compare_by]
    table = table[plot_cols]
    try:
        table.loc[:, compare_by] = table.loc[:, compare_by].astype(float)
    except ValueError:
        pass
    for t in _LOC['pval_thresholds']:
        table[f'nm {t}'] = table[f'px (p <{t})'] * table[f'mean (p <{t})']
        table[f'nmm {t}'] = table[f'px (p <{t})'] * table[f'mean (p <{t})'] ** 2
        table[f'nss {t}'] = table[f'px (p <{t})'] * table[f'std (p <{t})'] ** 2
    table = table.groupby(['structure', compare_by, ]).sum(min_count=1)
    table2 = pd.DataFrame()
    for t in _LOC['pval_thresholds']:
        table2[f'mean (p <{t})'] = table[f'nm {t}'] / table[f'px (p <{t})']
        table[f'Ess {t}'] = (table[f'nss {t}'] + table[f'nmm {t}']) \
            / table[f'px (p <{t})'] - table2[f'mean (p <{t})'] ** 2
        table2[f'std (p <{t})'] = np.sqrt(table[f'Ess {t}'])
        table2[f'px (p <{t})'] = table[f'px (p <{t})']
    table2.to_csv(save_path, sep='\t')
