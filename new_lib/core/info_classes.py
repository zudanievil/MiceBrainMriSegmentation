"""
This module has utility classes that represent abstract concepts
like research projects, masks, etc in a useful way
"""
import yaml
import shutil
import urllib.request
import pathlib
import dataclasses
import numpy as np
import xml.etree.ElementTree as et
import PIL

_LOC = {
    'download_slice_ids': 'http://api.brain-map.org/api/v2/data/query.xml?criteria=model::AtlasImage,rma::criteria,'
                          '[annotated$eqtrue],atlas_data_set(atlases[id$eq{atlas_id}]),'
                          'rma::options[order$eq%27sub_images.section_number%27][num_rows$eqall]',
    'download_svg': 'http://api.brain-map.org/api/v2/svg_download/{svg_id}?groups={svg_groups}',
    'download_ontology': 'http://api.brain-map.org/api/v2/structure_graph_download/{atlas_id}.xml',
    }


class OntologyFolderInfo:
    __slots__ = '_folder'

    def __init__(self, folder: 'pathlib.Path or str'):
        if isinstance(folder, self.__class__):
            self._folder = folder._folder
        else:
            self._folder = pathlib.Path(folder)

    def folder(self):
        return self._folder

    def onts_folder(self):
        return self._folder / 'onts'

    def svgs_folder(self):
        return self._folder / 'svgs'

    def ontology_info(self, frame: str) -> 'OntologyInfo':
        return OntologyInfo(self, frame)

    def write(self) -> 'files':
        self._folder.mkdir(exist_ok=False, parents=True)
        for subfolder in ('onts', 'svgs'):
            (self._folder / subfolder).mkdir()
        form_src = pathlib.Path(__name__).parent.parent / 'default_forms' / 'masks_download_form.yml'
        form_dst = self._folder / 'masks_download_form.yml'
        shutil.copy(form_src, form_dst)

    def download_slice_ids_table(self) -> 'files':
        pass

    def download_svgs(self) -> 'files':
        pass

    def download_default_ont(self) -> 'files':
        pass

    def ontology_folder_specifications(self) -> dict:
        pass

    def __iter__(self) -> 'collections.Iterable[OntologyInfo]':
        for o in (self.onts_folder() / 'onts').iterdir():
            f = o.with_suffix('').name
            if f != 'default':
                yield OntologyInfo(self, f)


class OntologyInfo:
    __slots__ = '_folder_info', '_frame'

    def __init__(self, folder_info: 'OntologyFolderInfo, pathlib.Path or str', frame: str):
        self._folder_info = OntologyFolderInfo(folder_info)
        self._frame = frame

    def folder_info(self) -> 'OntologyFolderInfo':
        return self._folder_info

    def frame(self) -> str:
        return self._frame

    def mask_path(self, mask_name: str = 'Root') -> 'pathlib.Path or None':
        """
        Performs linear search in ontology tree.
        :var mask_name must match exactly
        :returns pathlib.Path if structure is found and has 'filename' attribute, else None
        """
        rt = self.tree().getroot()
        for s in rt.iter('structure'):
            if s.attrib['name'] == mask_name:
                fn = s.attrib['filename']
                return pathlib.Path(fn) if fn else None

    def open_mask(self, mask_name: str = 'Root') -> np.ndarray:
        """"""
        path = self.mask_path(mask_name)
        if path:
            mask = PIL.Image.open(path)
            mask = np.array(mask.getdata(), dtype=np.uint8).reshape((mask.size[1], mask.size[0]))
            return mask > 127

    def tree_path(self) -> pathlib.Path:
        return self._folder_info.onts_folder() / (self._frame + '.xml')

    def tree(self):
        return et.parse(self.tree_path())

    def svg_path(self) -> pathlib.Path:
        return self._folder_info.onts_folder() / (self._frame + '.svg')

    def svg(self):
        return et.parse(self.svg_path())

    def default_tree_path(self) -> pathlib.Path:
        return self._folder_info.onts_folder() / 'default.xml'

    def default_tree(self):
        return et.parse(self.default_tree_path())


class ImageFolderInfo:
    __slots__ = '_folder'

    def __init__(self, folder: 'pathlib.Path or str'):
        if isinstance(folder, self.__class__):
            self._folder = folder._folder
        else:
            self._folder = pathlib.Path(folder)

    def folder(self):
        return self._folder

    def image_folder(self):
        return self._folder / 'img'

    def metadata_folder(self):
        return self._folder / 'meta'

    def cropped_image_folder(self):
        return self._folder / 'img_cropped'

    def pre_metadata_folder(self):
        return self._folder / 'pre_meta'

    def raw_image_folder(self):
        return self._folder / 'img_raw'

    def write(self):
        pass

    def image_folder_specifications(self) -> dict:
        pass

    def __iter__(self) -> 'collections.Iterable[ImageInfo]':
        for p in self.image_folder().iterdir():
            n = p.with_suffix('').name
            yield ImageInfo(self, n)


class ImageInfo:
    __slots__ = '_folder_info', '_name'

    def __init__(self, folder_info: 'ImageFolderInfo or pathlib.Path or str', name: str):
        self._folder_info = ImageFolderInfo(folder_info)
        self._name = name

    def name(self) -> str:
        return self._name

    def folder_info(self) -> 'ImageFolderInfo':
        return self._folder_info

    def raw_image_path(self, suffix='.img') -> pathlib.Path:
        return self._folder_info.raw_image_folder() / (self._name + suffix)

    def raw_image(self, dtype='<i4', shape=(256, 256)) -> np.ndarray:
        return np.fromfile(self.raw_image_path(), dtype=dtype).reshape(shape)

    def image_path(self) -> pathlib.Path:
        return self._folder_info.image_folder() / (self._name + '.npy')

    def image(self) -> np.ndarray:
        return np.load(self.image_path())

    def cropped_image_path(self) -> pathlib.Path:
        return self._folder_info.cropped_image_folder() / (self._name + '.npy')

    def cropped_image(self) -> np.ndarray:
        return np.load(self.cropped_image_path())

    def metadata_path(self) -> pathlib.Path:
        return self._folder_info.metadata_folder() / (self._name + '.yml')

    def metadata(self) -> dict:
        with self.metadata_path().open('rt') as f:
            m = yaml.safe_load(f.read())
        return m


class SegmentationResultFolderInfo:
    __slots__ = '_image_folder_info', '_ontology_folder_info', '_folder'

    def __init__(self, image_folder_info, ontology_folder_info, folder):
        self._image_folder_info = ImageFolderInfo(image_folder_info)
        self._ontology_folder_info = OntologyFolderInfo(ontology_folder_info)
        self._folder = pathlib.Path(folder)

    @classmethod
    def read(cls, folder):
        pass

    def spec_folder(self):
        return self._folder / 'spec'

    def structure_list_path(self):
        return self.spec_folder() / 'structures.txt'

    def batches_path(self):
        return self.spec_folder() / 'batches.txt'

    def specification_path(self) -> pathlib.Path:
        return self.spec_folder() / 'specification_form.yml'

    def mask_permutation_path(self) -> pathlib.Path:
        return self.spec_folder() / 'mask_permutation.py'

    def mask_permutation(self) -> callable:
        p = self.mask_permutation_path()
        if p.exists():
            with p.open('rt') as f:
                exec(f.read())
            return locals()['mask_permutation']
        else:
            return lambda x, *y: x

    def table_folder(self):
        return self.spec_folder() / 'tables'

    def plot_folder(self):
        return self.spec_folder() / 'plots'


# @dataclasses.dataclass(frozen=True)
# class MasksInfo:
#     """helper class that is a simple interface with a masks folder"""
#     folder: pathlib.Path = None
#
#     @classmethod
#     def new(cls, folder: 'pathlib.Path or str') -> ('MasksInfo', 'files'):
#         if isinstance(folder, str):
#             folder = pathlib.Path(folder)
#         folder.mkdir(exist_ok=False)
#         for subfolder in {'onts', 'svgs'}:
#             (folder / subfolder).mkdir()
#         form_src = pathlib.Path(__name__) / 'default_forms' / 'masks_download_form.yml'
#         form_dst = folder / 'masks_download_form.yml'
#         shutil.copy(form_src, form_dst)
#         return cls(folder)
#
#     def download_slice_ids_table(self) -> 'files':
#         import pandas as pd
#         save_path = self.folder / 'slice_id_table.txt'
#         with (self.folder / 'masks_download_form.yml').open('rt') as f:
#             download_info = yaml.safe_load(f.read())
#         url = _LOC['download_slice_ids'].format(atlas_id=download_info['atlas_id'])
#         urllib.request.urlretrieve(url, save_path)
#         rt = et.parse(save_path, parser=et.XMLParser(encoding='utf-8')).getroot()
#         slice_ids = []
#         for node in rt.iter(tag='id'):
#             slice_ids.append(int(node.text))
#         slice_ids.reverse()
#         slice_ids = np.array(slice_ids)
#         coords = np.linspace(download_info['slice_coordinates']['start'],
#                              download_info['slice_coordinates']['stop'], len(slice_ids))
#         t = pd.DataFrame()
#         t['ids'] = slice_ids
#         t['coo'] = coords
#         t.to_csv(save_path, sep='\t', index=False)
#         return t
#
#     def download_svgs(self) -> 'files':
#         with (self.folder / 'masks_download_form.yml').open('rt') as f:
#             download_info = yaml.safe_load(f.read())
#         for svg_name in download_info['svg_names_and_ids']:
#             svg_id = download_info['svg_names_and_ids'][svg_name]
#             svg_path = self.folder / 'svgs' / (svg_name + '.svg')
#             url = _LOC['download_svg'].format(svg_id=svg_id, svg_groups=download_info['atlas_svg_groups'])
#             urllib.request.urlretrieve(url, str(svg_path))
#             print(f'atlas: {download_info["atlas_id"]} slice: {svg_id} saved to: {svg_path}')
#
#     def download_default_ont(self) -> 'files':
#         default_ont_path = self.folder / 'onts' / 'default.xml'
#         with (self.folder / 'masks_download_form.yml').open('rt') as f:
#             download_info = yaml.safe_load(f.read())
#         url = _LOC['download_ontology'].format(atlas_id=download_info['atlas_id'])
#         urllib.request.urlretrieve(url, default_ont_path)
#         self._refactor_ontology(default_ont_path, default_ont_path)
#
#     @staticmethod
#     def _refactor_default_ont(load: pathlib.Path, save: pathlib.Path, skip: tuple = ()) -> 'files':
#         def recursively_add_structures(new_rt: et.Element, old_rt: et.Element, lvl: int):
#             for old_elem in old_rt:
#                 if old_elem[4].text.strip() in skip:
#                     break
#                 new_elem = new_rt.makeelement('structure', {
#                     'name': old_elem[4].text.strip('"').title().replace(',', ''),
#                     'acronym': old_elem[3].text.strip(' "'),
#                     'id': old_elem[0].text,
#                     'level': str(lvl),
#                     'filename': '',
#                 })
#                 new_rt.append(new_elem)
#                 try:
#                     if old_elem[10].tag == 'children':
#                         recursively_add_structures(new_elem, old_elem[10], lvl + 1)
#                     else:
#                         print(new_elem.attrib['name'])
#                 except IndexError:
#                     pass
#         old_rt = et.parse(load).getroot()
#         new_rt = et.Element('root')
#         recursively_add_structures(new_rt, old_rt, lvl=0)
#         et.ElementTree(new_rt[0]).write(save, encoding='utf8')
#
#     def list_frames(self) -> 'list[str]':
#         """returns a list of .svg file names"""
#         result = []
#         for svg_path in self.folder.iterdir():
#             result.append(svg_path.with_suffix('').name)
#         return result
#
#     def masks_are_rendered(self) -> bool:
#         """checks if folders with masks corresponding to .svg files are all present"""
#         frames = self.list_frames()
#         for f in frames:
#             if not ((self.folder / f).exists() and (self.folder / 'onts' / (f + '.xml'))):
#                 return False
#             return True
#
#     def ont(self, frame: str) -> 'OntologyInfo':
#         return OntologyInfo(self, frame=frame)
#
#     def __iter__(self):
#         """iterator that returns onts one by one"""
#         return self
#
#     def __next__(self) -> 'OntologyInfo':
#         frames = self.list_frames()
#         for f in frames:
#             return self.ont(f)
#
#
# @dataclasses.dataclass(frozen=True)
# class ProjectInfo:
#     """
#     Helper class that is a simple interface with a project.
#     use .write() method to create
#     """
#     name: str = None
#     map_file_path: pathlib.Path = None
#     masks_folder: pathlib.Path = None
#     raw_image_folder: pathlib.Path = None
#     image_folder: pathlib.Path = None
#     cropped_image_folder: pathlib.Path = None
#     pre_metadata_folder: pathlib.Path = None
#     metadata_folder: pathlib.Path = None
#     result_folder: pathlib.Path = None
#     specification_folder: pathlib.Path = None
#
#
#     @classmethod
#     def new_single_root_project(cls, root_folder: 'pathlib.Path or str',
#                                 masks_folder: 'pathlib.Path or str' = None, ) -> ('ProjectInfo', 'files'):
#         """
#         Create new instance, where all folders and files are in root_folder.
#         Use .create_folders() and .write_map() to create the objects in file system
#         :var root_folder: path to the folder where all the new project files will be stored
#         :var masks_folder: optional instance of a MasksInfo associated with project
#         """
#         root_folder = pathlib.Path(root_folder)
#         return cls(
#             name=root_folder.name,
#             map_file_path=root_folder / 'project_map.yml',
#             masks_folder=masks_folder,
#             raw_image_folder=root_folder / 'raw_images',
#             image_folder=root_folder / 'images',
#             cropped_image_folder=root_folder / 'cropped_images',
#             pre_metadata_folder=root_folder / 'pre_metadata',
#             metadata_folder=root_folder / 'metadata',
#             result_folder=root_folder / 'results',
#             specification_folder=root_folder / 'specs',
#             )
#
#     def create_folders(self, exist_ok=False):
#         """
#         Creates a folder for every specified field, which name ends with '_folder'.
#         """
#         folders = []
#         for k in self.__dict__:
#             v = self.__dict__[k]
#             if isinstance(v, pathlib.Path) and k.endswith('_folder'):
#                 folders.append(v)
#         if not exist_ok:
#             for f in folders:
#                 if f.exists():
#                     raise FileExistsError(f)
#         for f in folders:
#             f.mkdir(exist_ok=True, parents=True)
#
#     def write_map(self, exist_ok=False):
#         """writes map file"""
#         if self.map_file_path.exists() and not exist_ok:
#             raise FileExistsError(self.map_file_path)
#         proj_map = {}
#         for k, v in self.__dict__.items():
#             proj_map[k] = str(v) if isinstance(v, pathlib.Path) else v
#         proj_map.pop('name')
#         proj_map.pop('map_file_path')
#         with self.map_file_path.open('wt') as f:
#             yaml.safe_dump(proj_map, f)
#
#     @classmethod
#     def read(cls, map_file_path__or_folder: 'pathlib.Path or str') -> 'ProjectInfo':
#         """:var map_file_path__or_folder: path to map file or a folder with project_map.yml file"""
#         path = pathlib.Path(map_file_path__or_folder)
#         name = path.with_suffix('').name
#         if path.is_dir():
#             path /= 'project_map.yml'
#         with path.open('rt') as f:
#             kwargs = yaml.safe_load(f.read())
#         kwargs['name'] = name
#         kwargs['map_file_path'] = path
#         for k, v in kwargs.items():
#             kwargs[k] = pathlib.Path(v) if (k.endswith(('_folder', '_path')) and v) else v
#         return cls(**kwargs)
#
#
# @dataclasses.dataclass(frozen=True)
# class ImageInfo:
#     """
#     helper class that is a simple interface with an image
#     :parameter name_fields: it is assumed that name of each image consists
#     of uderscore-separated values that describe the image.
#     this tuple contains keys for the values in the name.
#     """
#     project: ProjectInfo = None
#     name: str = None
#     name_fields: 'tuple[str]' = ('group', 'animal', 'hour', 'frame')
#
#     def meta_path(self):
#         return self.project.metadata_folder / (self.name + '.yml')
#
#     def meta(self):
#         path = self.meta_path()
#         with path.open('rt') as f:
#             meta = yaml.safe_load(f.read())
#         return meta
#
#     def image_path(self):
#         return self.project.image_folder / (self.name + '.npy')
#
#     def image(self) -> np.ndarray:
#         return np.load(self.image_path(), fix_imports=False)
#
#     def cropped_image_path(self):
#         return self.project.cropped_image_folder / (self.name + '.npy')
#
#     def cropped_image(self):
#         return np.load(self.cropped_image_path(), fix_imports=False)
#
#     def name_pieces(self) -> dict:
#         return {k: v for k, v in zip(self.name_fields, self.name.split('_'))}
#
#
# @dataclasses.dataclass(frozen=True)
# class OntologyInfo:
#     """
#     helper class that is a simple interface with an Ontology
#     """
#     masks_info: MasksInfo = None
#     frame: str = ''
#
#     def tree_path(self) -> pathlib.Path:
#         return self.masks_info.folder / 'onts' / (self.frame + '.xml')
#
#     def tree(self):
#         pass
#
#     def svg_path(self) -> pathlib.Path:
#         return self.masks_info.folder / 'svgs' / (self.frame + '.svg')
#
#     def svg(self):
#         pass
#
#     def default_tree_path(self) -> pathlib.Path:
#         return self.masks_info.folder / 'onts' / 'default.xml'
#
#     def default_tree(self):
#         pass
#
#     def mask_path(self, mask_name='Root'):
#         """
#         loads ontology tree and searches for the mask path (linear search)
#         mask_name must match exactly the name in the ontology tree
#         :returns path to the corresponding mask or None if mask not found in the tree
#         """
#         raise NotImplementedError
#
#     def mask(self, mask_name='Root'):
#         """
#         :returns boolean mask or None
#         """
#         raise NotImplementedError
#         path = self.mask_path(mask_name)
#         if path:
#             mask = PIL.Image.open(path)
#             mask = np.array(mask.getdata(), dtype=np.uint8).reshape((mask.size[1], mask.size[0]))
#             return mask > 127
#

