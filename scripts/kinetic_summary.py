import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_some_kinetic_parameters(csv_path, save_path):  # TODO: split, to ignore_lib
    assert csv_path != save_path
    t1 = pd.read_csv(csv_path, sep='\t')
    t1.drop(columns=['is_ref', 'reference', 'group',
                     'mean (p <0.01)', 'mean (p <0.05)',
                     'std (p <0.05)', 'std (p <1)', 'std (p <0.01)',
                     'px (p <0.01)'], inplace=True)
    t1['mean (p <1)'] *= t1['px (p <1)']
    t1 = t1.groupby(['structure', 'hour', 'animal']).sum(min_count=1)
    t1['mean (p <1)'] /= t1['px (p <1)']
    # here we correct for missing values
    t1 = t1.unstack(level='animal')
    for c in ('mean (p <1)', 'px (p <1)'):
        m = t1.loc[:, c].to_numpy()
        mm = np.nanmean(m, axis=1, keepdims=True)
        mm = np.tile(mm, (1, m.shape[1]))
        m = np.where(np.isnan(m), mm, m)
        t1.loc[:, c] = m
        del m, mm
    t1['px (p <0.05)'].replace(np.nan, 0.0, inplace=True)
    t1 = t1.stack(level=['animal'])
    # =====
    t1 = t1.unstack(level=['hour'])
    hours = t1.columns.unique('hour').to_list()
    hours.sort()
    for i in range(len(hours) - 1):
        t1['d mean', hours[i+1]] = t1['mean (p <1)', hours[i+1]] - t1['mean (p <1)', hours[i]]
    t2_columns = ['S_max', 't(S_max)', 'V_max', 't(V_max)']
    t2 = pd.DataFrame(index=t1.index, columns=t2_columns)
    t2['S_max'] = t1['mean (p <1)'].to_numpy().max(axis=1)
    t2['t(S_max)'] = np.array(hours)[t1['mean (p <1)'].to_numpy().argmax(axis=1)]
    t2['V_max'] = t1['d mean'].to_numpy().max(axis=1)
    t2['t(V_max)'] = np.array(hours[1:])[t1['d mean'].to_numpy().argmax(axis=1)]
    mask = np.zeros(t1['px (p <1)'].shape, dtype=np.int)
    mask[np.arange(len(mask)), t1['mean (p <1)'].to_numpy().argmax(axis=1)] = 1
    m = (t1['px (p <0.05)']/t1['px (p <1)']).to_numpy()
    t2['%sign pix'] = m[mask > 0]*100
    del t1, m
    for column in t2_columns:
        t2[f'std( {column})'] = np.nan
    t2 = t2.unstack(level='animal')
    for column in t2_columns:
        t2[f'std( {column})'] = np.tile(t2[column].std(axis=1).to_numpy(), (6, 1)).T
    t2 = t2.stack(level='animal')
    t2.to_csv(save_path, sep='\t')


if __name__ == '__main__':
    for g in ('c57bl', 'cgtg'):
        lp = f"C:/Users/user/Desktop/new_segm/CgTg_{g}/results_left.txt"
        sp = f"C:/Users/user/Desktop/new_segm/CgTg_{g}/results_left_kinetic.txt"
        get_some_kinetic_parameters(lp, sp)
