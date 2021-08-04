"""

"""


import pandas
from ..core import info_classes


def table_of_images(image_folder_info: info_classes.ImageFolderInfo) -> pandas.DataFrame:
    table = []
    columns = ('name', *image_folder_info.specification()['file_name_fields'])
    for image_info in image_folder_info:
        name = image_info.name()
        column_values = (name, *name.split('_'))
        table.append(column_values)
    table = pandas.DataFrame(table, columns=columns)
    return table


def split_to_groups(table: pandas.DataFrame, spec: dict) -> pandas.DataFrame:
    # new_index = spec['compare_by'] + spec['match_by'] + spec['batch_by']
    new_index = table.columns.to_list()
    new_index.remove('name')
    table = table.set_index(new_index)[['name', ]].unstack(spec['batch_by'][0])
    reference_value = str(spec['comparison_reference_value'])
    table.sort_index(axis=0, inplace=True)
    table.columns.rename('is_reference', level=0, inplace=True)
    new_index = spec['compare_by'] + spec['match_by']
    table.reset_index(inplace=True)
    table.set_index(new_index, inplace=True)
    table = table[['name']]

    ref_index = table.index[table.index.get_loc(reference_value)]
    ref_table = table.loc[ref_index].rename(columns={'name': True})
    ref_table = ref_table.loc[reference_value]
    table = table.drop(index=ref_index)
    table.rename(columns={'name': False}, inplace=True)
    idx = table.index.get_level_values(0).unique()
    chunks = []
    for i in idx:
        x = table.loc[i]
        y = ref_table.loc[x.index]
        chunks.append(pandas.concat((x, y), axis=1))
    table = pandas.concat(chunks, axis=0)
    table.reset_index(drop=True, inplace=True)
    return table


def main(segmentation_result_folder_info: info_classes.segmentation_result_folder_info_like):
    rf = info_classes.SegmentationResultFolderInfo.read(segmentation_result_folder_info)
    image_folder_info = rf.image_folder_info()
    spec = rf.specification()['batching']
    t = table_of_images(image_folder_info)
    t = split_to_groups(t, spec)
    t.to_csv(rf.batches_path(), sep='\t', na_rep='NA', index=True)
