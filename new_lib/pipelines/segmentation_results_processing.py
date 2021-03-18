import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..core import info_classes
from ..utils import lang_utils


def main(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like,
         structure_list_for_significance_table=None):
    srfi = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)
    refactor_summary(srfi)
    make_kinetics_table(srfi)
    make_significance_table(srfi, structure_list_for_significance_table)
    plot_segmentation_results(srfi)


# ==================================================================
def refactor_summary(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like):
    """
    Use "compare_by" and pval thresholds to create a table with
    anatomical_structure/(mean_intensity, std, pvalue) axes
    ('segm_result.txt' -> 'segm_result_refactored.txt').
    The produced table is much more human-readable.
    """
    srfi = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)
    load_path = srfi.table_folder() / 'segm_result.txt'
    save_path = srfi.table_folder() / 'segm_result_refactored.txt'

    spec = srfi.specification()
    compare_by = spec['batching']['compare_by'][0]
    pval_thresholds = spec['comparison']['pval_thresholds']
    del spec

    table = pd.read_csv(load_path, sep='\t')
    plot_cols = table.columns[[c.startswith(('mean', 'std', 'px')) for c in table.columns]].to_list()
    plot_cols += ['structure', compare_by]
    table = table[plot_cols]
    try:
        table.loc[:, compare_by] = table.loc[:, compare_by].astype(float)
    except ValueError:
        pass

    for t in pval_thresholds:
        table[f'nm {t}'] = table[f'px (p <{t})'] * table[f'mean (p <{t})']
        table[f'nmm {t}'] = table[f'px (p <{t})'] * table[f'mean (p <{t})'] ** 2
        table[f'nss {t}'] = table[f'px (p <{t})'] * table[f'std (p <{t})'] ** 2
    table = table.groupby(['structure', compare_by, ]).sum(min_count=1)

    table2 = pd.DataFrame()
    for t in pval_thresholds:
        table2[f'mean (p <{t})'] = table[f'nm {t}'] / table[f'px (p <{t})']
        table[f'Ess {t}'] = (table[f'nss {t}'] + table[f'nmm {t}']) \
                            / table[f'px (p <{t})'] - table2[f'mean (p <{t})'] ** 2
        table2[f'std (p <{t})'] = np.sqrt(table[f'Ess {t}'])
        table2[f'px (p <{t})'] = table[f'px (p <{t})']
    table2.to_csv(save_path, sep='\t')


# ============================================================
def make_kinetics_table(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like):
    warnings.warn(message='make_kinetics_table subpipeline have not been generalized '
                          'for arbitrary filename fields, please inspect code')
    srfi = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)
    load_path = srfi.table_folder() / 'segm_result.txt'
    save_path = srfi.table_folder() / 'kinetic_table.txt'

    t1 = pd.read_csv(load_path, sep='\t')
    t1.drop(columns=['is_ref', 'reference', 'group',
                     'mean (p <0.01)', 'mean (p <0.05)',
                     'std (p <0.05)', 'std (p <1)', 'std (p <0.01)',
                     'px (p <0.01)'], inplace=True)
    t1['mean (p <1)'] *= t1['px (p <1)']
    t1 = t1.groupby(['structure', 'hour', 'animal']).sum(min_count=1)  # todo
    t1['mean (p <1)'] /= t1['px (p <1)']

    # here we correct for missing values
    t1 = t1.unstack(level='animal')  # todo
    for c in ('mean (p <1)', 'px (p <1)'):
        m = t1.loc[:, c].to_numpy()
        mm = np.nanmean(m, axis=1, keepdims=True)
        mm = np.tile(mm, (1, m.shape[1]))
        m = np.where(np.isnan(m), mm, m)
        t1.loc[:, c] = m
        del m, mm
    t1['px (p <0.05)'].replace(np.nan, 0.0, inplace=True)
    t1 = t1.stack(level=['animal'])  # todo
    # =====
    t1 = t1.unstack(level=['hour'])  # todo
    hours = t1.columns.unique('hour').to_list()  # todo
    hours.sort()
    for i in range(len(hours) - 1):
        t1['d mean', hours[i + 1]] = t1['mean (p <1)', hours[i + 1]] - t1['mean (p <1)', hours[i]]
    t2_columns = ['S_max', 't(S_max)', 'V_max', 't(V_max)']
    t2 = pd.DataFrame(index=t1.index, columns=t2_columns)
    t2['S_max'] = t1['mean (p <1)'].to_numpy().max(axis=1)
    t2['t(S_max)'] = np.array(hours)[t1['mean (p <1)'].to_numpy().argmax(axis=1)]
    t2['V_max'] = t1['d mean'].to_numpy().max(axis=1)
    t2['t(V_max)'] = np.array(hours[1:])[t1['d mean'].to_numpy().argmax(axis=1)]
    mask = np.zeros(t1['px (p <1)'].shape, dtype=np.int)
    mask[np.arange(len(mask)), t1['mean (p <1)'].to_numpy().argmax(axis=1)] = 1
    m = (t1['px (p <0.05)'] / t1['px (p <1)']).to_numpy()
    t2['%sign pix'] = m[mask > 0] * 100
    del t1, m
    for column in t2_columns:
        t2[f'std( {column})'] = np.nan
    t2 = t2.unstack(level='animal')  # todo
    for column in t2_columns:
        t2[f'std( {column})'] = np.tile(t2[column].std(axis=1).to_numpy(), (6, 1)).T
    t2 = t2.stack(level='animal')  # todo
    t2.to_csv(save_path, sep='\t')


# ========================================================================================
def make_significance_table(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like,
                            structure_list: 'list[str]'):
    warnings.warn(message='make_significance_table subpipeline have not been generalized '
                          'for arbitrary filename fields and arbitrary brain structures, please inspect code')
    srfi = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)
    load_path = srfi.table_folder() / 'kinetic_table.txt'
    save_path = srfi.table_folder() / 'significance_table.txt'

    # put result into a table
    index_t = pd.DataFrame([e.attrib for e in structure_list])
    index_t.drop(columns=['filename', 'id'], inplace=True)
    index_t.rename(columns={'name': 'structure'}, inplace=True)
    index_t.set_index('structure', inplace=True)

    # load kinetic table, extract useful parts
    t = pd.read_csv(load_path, sep='\t')
    sign_pix_t = t.pivot(index='structure', columns='animal', values='%sign pix')  # todo
    s_max_mob = t[t['structure'] == 'Main Olfactory Bulb']  # todo
    s_max_mob = s_max_mob.set_index('animal')['S_max']  # todo
    s_max_mob.name = 'S_max Main Olfactory Bulb'  # todo
    v_max_on = t[t['structure'] == 'Olfactory Nerve']  # todo
    v_max_on = v_max_on.set_index('animal')['V_max']  # todo
    v_max_on.name = 'V_max Olfactory Nerve'  # todo
    # join tables
    t2 = pd.concat([index_t, sign_pix_t], axis=1, join='inner').T
    t2 = pd.concat([t2, v_max_on, s_max_mob], axis=1)
    t2.to_csv(save_path, sep='\t')


# ==================================================================
def plot_segmentation_results(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like,
                              save_plots_with_segmentation_images: bool = True):
    srfi = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)
    load_path = srfi.table_folder() / 'segm_result_refactored.txt'
    plot_folder = srfi.plot_folder()

    spec = srfi.specification()
    compare_by = spec['batching']['compare_by'][0]
    pval_thresholds = spec['comparison']['pval_thresholds']
    plot_spec = spec['summary_plot']

    table = pd.read_csv(load_path, sep='\t',
                        index_col=['structure', compare_by])
    structures = np.unique(table.index.get_level_values('structure'))
    for structure in structures:  # todo: add tqdm
        save_name = structure.replace('/', '_')
        if save_plots_with_segmentation_images:
            save_path = lang_utils.find_file(save_name, plot_folder).parent
        else:
            save_path = plot_folder
        save_path /= (save_name + ' plot.png')

        data = table.loc[structure]
        fig, axs = plt.subplots(1, 1, **plot_spec['figure_kwargs'])
        for i, t in enumerate(pval_thresholds):
            axs.errorbar(data.index - 0.3 * i, data[f'mean (p <{t})'], yerr=data[f'std (p <{t})'],
                         **plot_spec['line_kwargs'][i])
        axs.set_xticks(data.index)
        axs.grid()
        axs.legend(loc='upper right')
        fig.savefig(save_path, **plot_spec['savefig_kwargs'])
        plt.close(fig)
        print(datetime.datetime.now(), save_path)
