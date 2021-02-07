"""
utility functions for common downloads,
other modules' configuration and so on
"""

import pathlib
import urllib.request

import numpy as np
import yaml

from . import pattern_utils, transform_utils

_LOC = dict()
_GLOB = dict()
_IMPORTS_THAT_NEED_CONFIG = (pattern_utils, transform_utils)


def new_project_folder(project_folder: pathlib.Path) -> None:
    project_folder.mkdir(exist_ok=False)
    for subfolder in _LOC['project_subfolders']:
        (project_folder / subfolder).mkdir()
    _create_default_project_spec(project_folder)


def _create_default_project_spec(project_folder):
    raise NotImplementedError


def new_masks_folder(folder: pathlib.Path) -> None:
    folder.mkdir(exist_ok=False)
    for subfolder in _LOC['masks_subfolders']:
        (folder / subfolder).mkdir()
    masks_info = _LOC['default_download_info']
    with (folder / 'download_info.yml').open('wt') as f:
        yaml.safe_dump(masks_info, f)


def download_default_ontology(masks_folder: pathlib.Path) -> None:
    default_ont_path = masks_folder / 'onts' / 'default.xml'
    with (masks_folder / 'download_info.yml').open('rt') as f:
        download_info = yaml.safe_load(f.read())
    url = _LOC['download_ontology'].format(atlas_id=download_info['atlas_id'])
    urllib.request.urlretrieve(url, str(default_ont_path))
    _refactor_ontology(str(default_ont_path), str(default_ont_path))


def download_slice_svgs(masks_folder: pathlib.Path) -> None:
    with (masks_folder / 'download_info.yml').open('rt') as f:
        download_info = yaml.safe_load(f.read())
    if not download_info['svg_names_and_ids']:
        print(f'"svg_names_and_ids" key in {masks_folder.as_posix()}/download_info.yml has no value. no svg downloaded')
    for svg_name in download_info['svg_names_and_ids']:
        svg_path = masks_folder / 'svgs' / (svg_name + '.svg')
        svg_id = download_info['svg_names_and_ids'][svg_name]
        url = _LOC['download_svg'].format(svg_id=svg_id, svg_groups=download_info['atlas_svg_groups'])
        urllib.request.urlretrieve(url, str(svg_path))
        print(f'atlas: {download_info["atlas_id"]} slice: {svg_id} saved to: {svg_path}')


def _refactor_ontology(load: pathlib.Path, save: pathlib.Path, skip: tuple = ()) -> None:
    import xml.etree.ElementTree as et

    def recursively_add_structures(new_rt: et.Element, old_rt: et.Element, lvl: int) -> None:
        for old_elem in old_rt:
            if old_elem[4].text.strip() in skip:
                break
            new_elem = new_rt.makeelement('structure', {
                'name': old_elem[4].text.strip('"').title().replace(',', ''),
                'acronym': old_elem[3].text.strip(' "'),
                'id': old_elem[0].text,
                'level': str(lvl),
                'filename': '',
                })
            new_rt.append(new_elem)
            try:
                if old_elem[10].tag == 'children':
                    recursively_add_structures(new_elem, old_elem[10], lvl+1)
                else:
                    print(new_elem.attrib['name'])
            except IndexError:
                pass

    old_rt = et.parse(load).getroot()
    new_rt = et.Element('root')
    recursively_add_structures(new_rt, old_rt, lvl=0)
    et.ElementTree(new_rt[0]).write(save, encoding='utf8')


def download_slice_ids_table(masks_folder: pathlib.Path) -> None:
    import pandas as pd
    import numpy as np
    import xml.etree.ElementTree as et
    save_path = masks_folder / 'slice_id_table.txt'
    with (masks_folder / 'download_info.yml').open('rt') as f:
        download_info = yaml.safe_load(f.read())
    url = _LOC['download_slice_ids'].format(atlas_id=download_info['atlas_id'])
    urllib.request.urlretrieve(url, save_path)
    rt = et.parse(save_path, parser=et.XMLParser(encoding='utf-8'))
    rt = rt.getroot()
    slice_ids = []
    for node in rt.iter(tag='id'):
        slice_ids.append(int(node.text))
    slice_ids.reverse()
    slice_ids = np.array(slice_ids)
    coords = np.linspace(download_info['atlas_first_slice_coord'],
                         download_info['atlas_last_slice_coord'], len(slice_ids))
    t = pd.DataFrame()
    t['ids'] = slice_ids
    t['coo'] = coords
    t.to_csv(save_path, sep='\t', index=False)
    return t


def compose_metadata(project: pathlib.Path, meta_src: pathlib.Path = None) -> None:
    def _parse_keys(generic_name: str, spec: dict) -> dict:
        meta = dict()
        for key in spec:
            key_file_name = generic_name + '_' + key + '.txt'
            dtype = np.dtype(spec[key]['dtype'])
            selector = np.array(spec[key]['selector'], dtype=np.int)
            val = np.fromfile(key_file_name, sep='\t', dtype=dtype)[selector].tolist()
            meta[key] = val if len(val) > 1 else val[0]
        return meta

    meta_src = meta_src or project / 'pre_meta'
    regex = pattern_utils.fstring_to_regex(_GLOB['file_naming_convention'])
    additional_keys = tuple(regex.groupindex)
    for image in (project / 'img').iterdir():
        meta = _parse_keys(generic_name=str(meta_src / image.stem),
                           spec=_LOC['metadata_keys'], )
        additional_values = regex.match(image.stem).groups()
        meta.update(zip(additional_keys, additional_values))
        save_to = project / 'meta' / (image.stem + '.yml')
        with save_to.open('wt') as f:
            yaml.safe_dump(meta, f)


def crop_and_resize_images(project: pathlib.Path, save_png_previews: bool = False) -> None:  # checked
    if save_png_previews:
        import matplotlib.pyplot as plt
    for image_path in (project / 'img').iterdir():
        image_name = image_path.stem
        meta_path = project / 'meta' / (image_name + '.yml')
        image = np.load(image_path, fix_imports=False)
        with meta_path.open('rt') as f:
            meta = yaml.safe_load(f)
        image = transform_utils.rotate(image, -meta['rotation'])
        frame = meta['frame']
        frame_shape = _GLOB['frame_shapes'][frame]
        left = transform_utils.bbox_crop(image, meta['lbbox'])
        right = transform_utils.bbox_crop(image, meta['rbbox'])
        s = frame_shape[0], frame_shape[1] // 2
        left = transform_utils.resize(left, s)
        right = transform_utils.resize(right, s)
        image = np.concatenate([left, right], axis=1)
        np.save(project / 'img_cropped' / image_name, image, fix_imports=False)
        if save_png_previews:
            preview_path = project / 'img_cropped' / (image_name+'.png')
            plt.imsave(preview_path, image, cmap='gray', format='png')
