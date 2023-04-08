from typing import Iterable, Union, List, Optional, Callable
from pathlib import Path

import pandas as pd
import numpy as np

from ..core import info_classes as IC


_STRUCTURE_KEY = "structure"


def group_by_cols_and_average(table, group_by, pval_thresholds):
    for t in pval_thresholds:
        table[f'nm {t}'] = table[f'px (p <{t})'] * table[f'mean (p <{t})']
        table[f'nmm {t}'] = table[f'px (p <{t})'] * table[f'mean (p <{t})'] ** 2
        table[f'nss {t}'] = table[f'px (p <{t})'] * table[f'std (p <{t})'] ** 2
    table = table.groupby(group_by).sum(min_count=1)
    table2 = pd.DataFrame()
    for t in pval_thresholds:
        table2[f'mean (p <{t})'] = table[f'nm {t}'] / table[f'px (p <{t})']
        table[f'Ess {t}'] = (table[f'nss {t}'] + table[f'nmm {t}']) / table[f'px (p <{t})'] - table2[
            f'mean (p <{t})'] ** 2
        table2[f'std (p <{t})'] = np.sqrt(table[f'Ess {t}'])
        table2[f'px (p <{t})'] = table[f'px (p <{t})']
    table2.reset_index(inplace=True)
    return table2


def add_result_name_columns(table: pd.DataFrame, result_namer, srfi) -> (pd.DataFrame, List[str]):
    """IN PLACE"""
    if isinstance(result_namer, str):
        table[result_namer] = srfi.folder().name
        col_names = [result_namer, ]
    else:
        cols = [result_namer(srfi.folder().name)] * len(table)
        cols = pd.DataFrame(cols, index=table.index)
        col_names = cols.columns.to_list()
        table = pd.concat((cols, table), axis="columns")
    return table, col_names


def take_structures(table: pd.DataFrame, structures):
    return table.set_index([_STRUCTURE_KEY]).loc[structures].reset_index()


def main(
        srfis: Iterable[IC.segmentation_result_folder_info_like],
        save_path: Union[Path, str],
        result_namer: Union[Callable[[str], dict], str] = "group",
        structures: Optional[List[str]] = None,
):
    """
    :param srfis: results for which you want to make a single summary.
    :param save_path: where to save the summary.
    :param result_namer: either a name for the column that will store result folder names or
    a callable that returns dict {column_name_1: value_1, ...} given the result folder name.
    :param structures: if present, create summary for these structures only.
    NB: you need to personally verify that such combination procedure makes sense for your results.
    NB: this function assumes that columns left in each result table are:
        "structure" + `group_by`, `match_by`, `batch_by` from result folder configurations,
        columns that start with "mean ", "std ", "px "



    """
    srfis = tuple(IC.SegmentationResultFolderInfo.read(f) for f in srfis)
    save_path = Path(save_path).resolve()
    added_col_names = []  # don't pay attention, this is for the type checker

    config = srfis[0].configuration()
    pval_thresholds: List[float] = config["comparison"]["pval_thresholds"]
    compare_by = config["batching"]["compare_by"]
    match_by = config["batching"]["match_by"]
    batch_by = config["batching"]["batch_by"]

    idx_cols = [_STRUCTURE_KEY] + compare_by + match_by + batch_by  # ['structure', 'hour', 'frame', 'animal']

    chuncks = []
    for srfi in srfis:
        load_path = srfi.table_folder() / "segm_result.txt"
        table = pd.read_csv(load_path, sep='\t')

        # select columns to keep, select columns to serve as index later on
        plot_cols = [c for c in table.columns if c.startswith(("mean ", "std ", "px "))]
        plot_cols += idx_cols
        table = table[plot_cols]

        table = take_structures(table, structures) if structures is not None else table
        table = group_by_cols_and_average(table, idx_cols, pval_thresholds)
        table, added_col_names = add_result_name_columns(table, result_namer, srfi)

        chuncks.append(table)
    table = pd.concat(chuncks, axis=0)

    i = added_col_names + idx_cols
    table.set_index(i, inplace=True)

    table.sort_index(axis=0, inplace=True)
    table.to_csv(save_path, sep='\t')

