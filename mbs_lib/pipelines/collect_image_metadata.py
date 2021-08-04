import pathlib
import yaml
import numpy

from ..core import info_classes


def read_metadata_chuncks(fname_start: pathlib.Path, metadata_keys: dict, not_found_list: list) -> dict:
    fname_start = str(fname_start)
    meta = dict()
    for k, v in metadata_keys.items():
        fname = f'{fname_start}_{k}.txt'
        dtype = numpy.dtype(v['dtype'])
        idx = numpy.array(v['indices'], dtype=numpy.int)
        try:
            chunck = numpy.fromfile(fname, sep='\t', dtype=dtype)[idx].tolist()
            meta[k] = chunck if len(chunck) > 1 else chunck[0]
        except FileNotFoundError:
            not_found_list.append(pathlib.Path(fname))
    return meta


def fname_gen(img_folder: info_classes.ImageFolderInfo):
    f0 = img_folder.raw_image_folder()
    f1 = img_folder.image_folder()
    f = f0 if f0.exists() else f1
    for p in f.iterdir():
        yield p.stem


def main(img_folder: info_classes.image_folder_info_like) -> 'list[pathlib.Path]':
    """
    :return: list with paths for the not found metadata chuncks, which can be empty
    """
    img_folder = info_classes.ImageFolderInfo(img_folder)
    img_folder_spec = img_folder.specification()
    fname_fields = img_folder_spec['file_name_fields']
    metadata_keys = img_folder_spec['metadata_keys']
    meta_source_folder = img_folder.pre_metadata_folder()
    meta_destination_folder = img_folder.metadata_folder()
    not_found_list = []
    for fname in fname_gen(img_folder):
        # print(fname)
        meta = read_metadata_chuncks(meta_source_folder / fname, metadata_keys, not_found_list)
        meta.update(zip(fname_fields, fname.split('_')))
        save_to = meta_destination_folder / (fname + '.yml')
        with save_to.open('wt') as f:
            yaml.safe_dump(meta, f)
    return not_found_list
