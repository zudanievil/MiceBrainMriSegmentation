
"""
utility functions for common downloads,
other modules' configuration and so on
"""

import pathlib
import urllib.request
import yaml
import numpy as np
from . import pattern_utils, transform_utils

_LOC = dict()
_GLOB = dict()
_IMPORTS_THAT_NEED_CONFIG = (pattern_utils, transform_utils)


def new_project_folder(project_folder: pathlib.Path) -> None:
    project_folder.mkdir(exist_ok=False)
    for subfolder in _LOC['project_subfolders']:
        (project_folder / subfolder).mkdir()


def new_masks_folder(folder: pathlib.Path) -> None:
    folder.mkdir(exist_ok=False)
    for subfolder in _LOC['masks_subfolders']:
        (folder / subfolder).mkdir()

    default_ont_path = folder / 'onts' / 'default.xml'
    url = _LOC['download_ontology'].format(atlas=_LOC['atlas_id'])
    urllib.request.urlretrieve(url, str(default_ont_path))
    _refactor_ontology(str(default_ont_path), str(default_ont_path))
    if not _LOC['svg_names_and_ids']:
        print('"svg_names_and_ids" config var is empty. no svg downloaded')
    for svg_name in _LOC['svg_names_and_ids']:
        svg_path = (folder / 'svgs' / svg_name).with_suffix('.svg')
        svg_id = _LOC['svg_names_and_ids'][svg_name]
        url = _LOC['download_svg'].format(id=svg_id, groups=_LOC['svg_groups'])
        urllib.request.urlretrieve(url, str(svg_path))
        print(f'atlas: {_LOC["atlas_id"]} slice: {svg_id} saved to: {svg_path}')


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


def fetch_slice_ids_table(save_path: pathlib.Path) -> None:
    import pandas as pd
    import numpy as np
    import xml.etree.ElementTree as et
    save_path = save_path.with_suffix('.txt')
    url = _LOC['download_slice_ids'].format(atlas=_LOC['atlas_id'])
    urllib.request.urlretrieve(url, save_path)
    rt = et.parse(save_path, parser=et.XMLParser(encoding='utf-8'))
    rt = rt.getroot()
    slice_ids = []
    for node in rt.iter(tag='id'):
        slice_ids.append(int(node.text))
    slice_ids.reverse()
    slice_ids = np.array(slice_ids)
    coords = np.linspace(_LOC['atlas_first_slice_coord'], _LOC['atlas_last_slice_coord'], len(slice_ids))
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
