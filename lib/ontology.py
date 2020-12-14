import sys
import pathlib
import datetime
import dataclasses
import xml.etree.ElementTree as et
import numpy as np
import subprocess

import PIL.Image

_LOC = dict()
_GLOB = dict()

File = type(None)
class EmptyMaskException(Exception): pass


@dataclasses.dataclass(frozen=True)
class Ontology:
    folder: pathlib.Path
    name: str

    @property
    def xml_path(self) -> pathlib.Path:
        return self.folder / 'onts' / (self.name + '.xml')

    @property
    def svg_path(self) -> pathlib.Path:
        return self.folder / 'svgs' / (self.name + '.svg')

    @property
    def xml_tree(self) -> et.ElementTree:
        return et.parse(self.xml_path, et.XMLParser(encoding='utf-8'))

    @property
    def svg_tree(self) -> et.ElementTree:
        return et.parse(self.svg_path, et.XMLParser(encoding='utf-8'))

    @property
    def default_xml_tree(self) -> et.ElementTree:
        return et.parse(self.folder / 'onts' / 'default.xml', et.XMLParser(encoding='utf-8'))


def load_mask(mask_path: pathlib.Path) -> np.ndarray:
    img = PIL.Image.open(mask_path.with_suffix('.png'))
    img = np.array(img.getdata(), dtype=np.uint8).reshape((img.size[1], img.size[0]))
    return img


def masks_generator(ont: Ontology) -> ('np.ndarray[bool]', dict):
    """
    :yields all structure masks and info for supplied ont
    If you send True to instance of this generator,
    it will skip substructures of a current structure
    make sure you understand generator.send() method before using this, though.
    """
    ont_root = ont.xml_tree.getroot()
    skip_subnodes = None
    prev_level = 0
    for node in ont_root.iter('structure'):
        if skip_subnodes:
            if int(node.attrib['level']) > prev_level:
                continue
        mask_path = ont.folder / ont.name / node.attrib['filename']
        mask = load_mask(mask_path)
        mask = mask > _LOC['mask_threshold']
        skip_subnodes = (yield mask, node.attrib)
        prev_level = int(node.attrib['level'])


class render_mask:
    """
    Static class, that simply aggregates functions for certain task.
    Main method is call(), this is what you typically need.
    See help(call) for details
    """
    @staticmethod
    def svg_from_ids(save_path: pathlib.Path, ont: Ontology, ids: 'set[str]') -> File:  # TODO: ids to set
        save_path = save_path.with_suffix('.svg')
        rt = ont.svg_tree.getroot()
        empty = True
        crop_id = _LOC['svg_crop_id']
        node_list = list(rt.iter())  # generator crashes if you iterate over tree while modifying it
        for node in node_list:
            if node.tag.endswith('namedview'):
                node.set('pagecolor', "#000000")
            try:
                if node.attrib['structure_id'] in ids:
                    node.set('style', 'stroke:none;fill:#ffffff;fill-opacity:1')
                    empty = False
                elif node.attrib['structure_id'] == crop_id:
                    node.set('style', 'stroke:none;fill:#000000;fill-opacity:0')
                else:
                    node_parent = find_xml_node_parent(node, rt)
                    node_parent.remove(node)
            except KeyError:
                if not node.tag.endswith(tuple(_LOC['allowed_svg_tag_tails'])):
                    node_parent = find_xml_node_parent(node, rt)
                    node_parent.remove(node)
        if empty:
            raise EmptyMaskException()
        et.ElementTree(rt).write(save_path, encoding='utf8')

    @staticmethod
    def svg_to_png(save_path: pathlib.Path, shape: (int, int), delete_src: bool = True) -> File:
        src_path = save_path.with_suffix('.svg')
        dst_path = save_path.with_suffix('.png')
        inkscape = pathlib.Path(_LOC['inkscape_exe'])
        cmd = _LOC['rendering_cmd']
        cmd = cmd.format(inkscape=inkscape, export_id=_LOC['svg_crop_id'],
                         src_path=src_path, dst_path=dst_path, height=shape[0], width=shape[1])
        with subprocess.Popen(cmd) as process:
            try:
                process.wait(10)  # after inkscape 1.0 update,
            # sometimes process does not exit, despite normal execution results
            except subprocess.TimeoutExpired:
                print(datetime.datetime.now(), 'Inkscape process timeout', file=sys.stderr)
                process.kill()
        if delete_src:
            src_path.unlink()

    @staticmethod
    def png_to_grayscale(src_path) -> File:
        src_path = src_path.with_suffix('.png')
        img = PIL.Image.open(src_path)
        img.load()
        img = img.convert('L')
        img.save(src_path)

    @staticmethod
    def get_structure_ids(node: et.Element) -> 'set[str]':
        if _LOC['include_substructures']:
            ids = []
            for subnode in node.iter():
                ids.append(subnode.attrib['id'])
        else:
            ids = [node.attrib['id'], ]
        return set(ids)

    @staticmethod
    def call(save_path: pathlib.Path, ont: Ontology, node: et.Element, mask_shape: (int, int)) -> File:
        """:raises EmptyMaskException if ids not found in svg"""
        cls = render_mask
        ids = cls.get_structure_ids(node)
        cls.svg_from_ids(save_path, ont, ids)
        cls.svg_to_png(save_path, mask_shape)
        cls.png_to_grayscale(save_path)


class prerender_masks:
    """
    Static class, that simply aggregates functions for certain task.
    Main method is call(), this is what you typically need.
    See help(call) for details
    """
    @staticmethod
    def assert_shapes_defined(ont_names: 'tuple[str]') -> None:
        for name in ont_names:
            if name not in _GLOB['frame_shapes']:
                raise AssertionError(f'Mask shape for {name} is missing. Set it in the config file\n'
                                     f'under \'global/frame_shapes\' keys')

    @staticmethod
    def assert_svgs_present(ont_names: 'tuple[str]', masks_folder: pathlib.Path) -> None:
        for name in ont_names:
            p = masks_folder / 'svgs' / (name + '.svg')
            if not p.exists():
                raise AssertionError(f'File {p} is missing')

    @staticmethod
    def assert_xmls_abscent(ont_names: 'tuple[str]', masks_folder: pathlib.Path) -> None:
        for name in ont_names:
            p = masks_folder / 'onts' / (name + '.xml')
            if p.exists():
                raise AssertionError(f'File {p} is present. Please delete or rename\n'
                                     f'excess .xml files in {masks_folder/"onts"}')

    @staticmethod
    def assert_path_is_short(masks_folder: pathlib.Path) -> None:
        if sys.platform.startswith('win') and len(str(masks_folder)) > 100:
            raise AssertionError('On windows long file paths (> 260 chars) are not allowed.'
                                 'Since masks are nested hierarchically in directories,'
                                 'it is best if \'masks_folder\' path is < 100 chars')

    @staticmethod
    def compose_mask_path(node: et.Element, node_parent: et.Element) -> pathlib.Path:
        if node_parent:
            p = pathlib.Path(node_parent.attrib['filename']).parent
        else:
            p = pathlib.Path('.')
        p = p / node.attrib['acronym'].replace('/', '_') / node.attrib['name'].replace('/', '_')
        return p

    @staticmethod
    def mask_is_ok(mask: np.ndarray) -> bool:
        if isinstance(mask, None.__class__):
            return False
        bmask = mask > _LOC['mask_threshold']
        return bmask.sum() > _LOC['min_mask_size']

    @staticmethod
    def prerendering_info() -> et.Element:
        metadata = et.Element('prerendering_info')
        metadata.attrib = {
            'include_substructures': str(_LOC['include_substructures']),
            'min_mask_size': str(_LOC['min_mask_size']),
            'mask_threshold': str(_LOC['mask_threshold']),
            'time': str(datetime.datetime.now()),
        }
        return metadata

    @staticmethod
    def del_structure_from_ont(node, node_parent) -> None:
        if node_parent:
            node_parent.remove(node)

    @staticmethod
    def call(masks_folder: pathlib.Path, ont_names: 'tuple[str]' = None) -> File:
        """
        for ont name in ont names takes
        {masks_folder}/onts/default.xml and {masks_folder}/svgs/{ont_name}.svg
        prerenders all masks found in svg.
        Saves modified ontology to {masks_folder}/onts/{ont_name}.xml,
        saves masks to {masks_folder}/{ont_name}.
        If ont_names = None, infers them.
        """
        cls = prerender_masks
        if ont_names:
            cls.assert_shapes_defined(ont_names)
        else:
            ont_names = tuple(_GLOB['frame_shapes'].keys())
        cls.assert_svgs_present(ont_names, masks_folder)
        cls.assert_xmls_abscent(ont_names, masks_folder)
        cls.assert_path_is_short(masks_folder)
        for ont_name in ont_names:
            ont = Ontology(masks_folder, ont_name)
            ont_root = ont.default_xml_tree.getroot()
            mask_shape = _GLOB['frame_shapes'][ont_name]
            print(datetime.datetime.now(), ' prerendering masks for: ', ont.xml_path)
            previous_mask_is_ok = True
            previous_level = 0
            node_list = list(ont_root.iter('structure'))
            for node in node_list:
                level = int(node.attrib['level'])
                if not previous_mask_is_ok and level > previous_level:
                    continue
                node_parent = find_xml_node_parent(node, ont_root)
                mask_path_relative = cls.compose_mask_path(node, node_parent)
                mask_path = masks_folder / ont_name / (str(mask_path_relative) + '.png')
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    render_mask.call(mask_path, ont, node, mask_shape)
                    mask = load_mask(mask_path)
                    mask_is_ok = cls.mask_is_ok(mask)
                    if mask_is_ok:
                        node.attrib['filename'] = mask_path_relative.as_posix()
                        print(datetime.datetime.now(), node.attrib['name'],)
                    else:
                        mask_path.unlink()
                        raise EmptyMaskException
                except EmptyMaskException:
                    mask_path.parent.rmdir()
                    mask_is_ok = False
                    if node_parent:
                        node_parent.remove(node)
                previous_level = level
                previous_mask_is_ok = mask_is_ok
            ont_root.append(cls.prerendering_info())
            et.ElementTree(ont_root).write(ont.xml_path, encoding='utf-8')
            print('success')
        print(datetime.datetime.now(), 'Finished')


def find_xml_node_parent(node: et.Element, tree_root: et.Element) -> 'et.Element or None':
    if node.attrib == tree_root.attrib:
        return None
    for potential_parent in tree_root.iter():
        if node in list(potential_parent):
            return potential_parent