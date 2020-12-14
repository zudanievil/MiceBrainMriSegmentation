import sys
import pathlib
import xml.etree.ElementTree as et
import numpy as np
import subprocess
import PIL.Image

_LOC = dict()
_GLOB = dict()

File = type(None)

class EmptyMaskException(Exception):
    pass


class Ontology:
    def __init__(self, mask_save_folder: pathlib.Path = None, ontology_path: pathlib.Path = None,
                 svg_path: pathlib.Path = None, xml_tree: et.ElementTree = None) -> None:
        self.mask_folder = mask_save_folder
        self.xml_path = ontology_path
        self.svg_path = svg_path
        self.xml_tree = xml_tree
        self.min_mask_size = _LOC['ontology.min_mask_size']
        self.mask_threshold = _LOC['ontology.mask_threshold']
        self.include_substructures = _LOC['ontology.include_substructures']
        mask_save_folder.mkdir(exist_ok=True)
        assert _LOC['ontology.mask_dir_structure'] in ('deep', 'wide')
        if _LOC['ontology.mask_dir_structure'] == 'deep':
            self.name_the_mask = self._name_the_mask_deep
            if sys.platform.startswith('win'):
                assert len(str(mask_save_folder)) <= 100, 'i recommend < 100 chars for \'mask_save_folder\' for '\
                                                          'windows\nand config[\'ontology.file_naming\']=\'deep\''
        elif _LOC['ontology.mask_dir_structure'] == 'wide':
            self.name_the_mask = self._name_the_mask_shallow
    
    def load(self, path: pathlib.Path = None) -> None:
        path = path or self.xml_path
        self.xml_tree = et.parse(path, et.XMLParser(encoding='utf-8'))
        
    def save(self, path: pathlib.Path = None) -> None:
        path = path or self.xml_path
        self.xml_tree.write(path, encoding='utf8')

    def find_all_intersections(self, bool_img: np.ndarray) -> (np.ndarray, dict):
        def _na():  # macro-like
            nonlocal node
            nonlocal prev_level
            nonlocal prev_not_found
            node.attrib['filename'] = 'NA'
            prev_level = level
            prev_not_found = True
        rt = self.xml_tree.getroot()
        prev_level = 0
        prev_not_found = False
        for node in rt.iter():
            level = int(node.attrib['level'])
            if prev_not_found and (level > prev_level):
                continue
            if node.attrib['filename'] == 'NA':
                _na()
                continue
            elif node.attrib['filename']:
                mask_path = self.mask_folder / node.attrib['filename']
                mask = self.load_mask(mask_path, bool_img.shape)
            else:
                filename = self.name_the_mask(rt, node)
                mask_path = self.mask_folder / filename
                mask_path.parent.mkdir(exist_ok=True)
                try:
                    mask = self.create_mask_for_ontology_node(node, self.svg_path, mask_path, bool_img.shape,
                                                              min_mask_size=self.min_mask_size,
                                                              mask_threshold=self.mask_threshold,
                                                              include_substructures=self.include_substructures)
                    node.attrib['filename'] = filename.as_posix()
                except EmptyMaskException:
                    _na()
                    continue
            yield mask > self.mask_threshold, node.attrib
            prev_not_found = False
            del mask

    @staticmethod
    def create_mask_for_ontology_node(node: et.Element, svg_path: pathlib.Path, save_path: pathlib.Path,
                                      render_resolution: (int, int), min_mask_size: int,
                                      mask_threshold: int, include_substructures: bool) -> np.ndarray:
        if include_substructures:
            ids = []
            for subnode in node.iter():
                ids.append(subnode.attrib['id'])
        else:
            ids = [node.attrib['id'], ]
        Ontology._create_new_svg_from_ids(svg_path, save_path, ids)
        Ontology._render_svg(save_path, save_path, render_resolution)
        mask = Ontology._png_to_mask(save_path, save_path)
        if np.sum(mask > mask_threshold) < min_mask_size:
            save_path.with_suffix('.png').unlink()  # unlink=delete
            raise EmptyMaskException()
        return mask

    @staticmethod
    def _create_new_svg_from_ids(load_path: pathlib.Path, save_path: pathlib.Path,
                                 structure_ids: list, crop_id: str = 'bbox') -> None:
        save_path = save_path.with_suffix('.svg')
        assert load_path != save_path
        rt = et.parse(load_path).getroot()
        crop_id = crop_id or rt.attrib['id']
        empty = True
        for node in rt.iter():
            if node.tag.endswith('namedview'):
                node.set('pagecolor', "#000000")
            for subnode in list(node):
                try:
                    if subnode.attrib['structure_id'] in structure_ids:
                        subnode.set('style', 'stroke:none;fill:#ffffff;fill-opacity:1')
                        empty = False
                    elif subnode.attrib['structure_id'] == crop_id:
                        subnode.set('style', 'stroke:none;fill:#000000;fill-opacity:0')
                    else:
                        node.remove(subnode)
                except KeyError:
                    if not subnode.tag.endswith(tuple(_LOC['svg.allowed_svg_tag_tails'])):
                        node.remove(subnode)
        if empty:
            raise EmptyMaskException()
        et.ElementTree(rt).write(save_path, encoding='utf8')

    @staticmethod
    def _render_svg(src_path: pathlib.Path, save_path: pathlib.Path,
                    resolution: (int, int), delete_src: bool = True, export_id: str = 'bbox') -> None:
        src_path = src_path.with_suffix('.svg')
        save_path = save_path.with_suffix('.png')
        inkscape = pathlib.Path(_LOC['svg.inkscape'])
        cmd = _LOC['svg.cmd']
        cmd = cmd.format(inkscape=inkscape, export_id=export_id,
                         src_path=src_path, save_path=save_path,
                         height=resolution[0], width=resolution[1])
        with subprocess.Popen(cmd) as process:
            process.wait(10)  # sometimes shell does not close after command execution on my computer W(0_0)W
            # it may be an interesting idea to render all masks beforehand and never call this during the segmentation
            # this would be much more in the spirit of the other scripts. too bad, i thought of this so late
        if delete_src:
            src_path.unlink()

    @staticmethod
    def _png_to_mask(src_path: pathlib.Path, save_path: pathlib.Path, delete_src: bool = True) -> np.ndarray:
        src_path = src_path.with_suffix('.png')
        save_path = save_path.with_suffix('.png')
        img = PIL.Image.open(src_path)
        img.load()
        img = img.convert('L')
        if delete_src:
            src_path.unlink()
        img.save(save_path)
        img = np.array(img.getdata(), dtype=np.uint8).reshape((img.size[1], img.size[0]))
        return img

    @staticmethod
    def load_mask(path: pathlib.Path, shape: "ignored" = None) -> np.ndarray:
        img = PIL.Image.open(path.with_suffix('.png'))
        img = np.array(img.getdata(), dtype=np.uint8).reshape((img.size[1], img.size[0]))
        return img

    @staticmethod
    def _name_the_mask_shallow(tree_root: et.Element, node: et.Element) -> str:
        p = pathlib.Path('./')
        p = p / node.attrib['acronym'].replace('/', '.') / node.attrib['name'].replace('/', '.')
        return p

    @staticmethod
    def _name_the_mask_deep(tree_root: et.Element, node: et.Element) -> pathlib.Path:
        parent = find_xml_node_parent(tree_root, node)
        if parent:
            p = pathlib.Path(parent.attrib['filename']).parent
        else:
            p = pathlib.Path('./')
        p = p / node.attrib['acronym'].replace('/', '.') / node.attrib['name'].replace('/', '.')
        return p


def find_xml_node_parent(tree_root: et.Element, node: et.Element) -> et.Element:
    for potential_parent in tree_root.iter():
        if node in list(potential_parent):
            return potential_parent


def prerender_masks(mask_folder: pathlib.Path) -> File: #TODO: separate mask search and mask rendering
    raise NotImplementedError

#TODO: rewrite in fp-style
