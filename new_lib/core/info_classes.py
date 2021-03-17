"""
This module has utility classes that represent abstract concepts
like research projects, masks, etc in a useful way
"""


import yaml
import shutil
import urllib.request
import pathlib
import numpy as np
import xml.etree.ElementTree as et
import PIL
import typing

_LOCALS = {
    'download_slice_ids': 'http://api.brain-map.org/api/v2/data/query.xml?criteria=model::AtlasImage,rma::criteria,'
                          '[annotated$eqtrue],atlas_data_set(atlases[id$eq{atlas_id}]),'
                          'rma::options[order$eq%27sub_images.section_number%27][num_rows$eqall]',
    'download_svg': 'http://api.brain-map.org/api/v2/svg_download/{svg_id}?groups={svg_groups}',
    'download_ontology': 'http://api.brain-map.org/api/v2/structure_graph_download/{atlas_id}.xml',
    }

path_like = typing.Union[pathlib.Path, str]
ontology_folder_info_like = typing.Union['OntologyFolderInfo', path_like]
ontology_info_like = typing.Union['OntologyInfo', path_like]
image_folder_info_like = typing.Union['ImageFolderInfo', path_like]
image_info_like = typing.Union['ImageInfo', path_like]
segmentation_result_folder_info_like = typing.Union['SegmentationResultFolderInfo', path_like]
image_info_iterator_type = typing.Iterator['ImageInfo']
ontology_info_iterator_type = typing.Iterator['OntologyInfo']
default_forms_folder = pathlib.Path(__file__).parent.parent / 'default_forms'


class OntologyFolderInfo:
    __slots__ = '_folder'

    def __init__(self, folder: ontology_folder_info_like):
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
        form_src = default_forms_folder / 'masks_download_form.yml'
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

    def __iter__(self) -> ontology_info_iterator_type:
        for o in (self.onts_folder() / 'onts').iterdir():
            f = o.with_suffix('').name
            if f != 'default':
                yield OntologyInfo(self, f)


class OntologyInfo:
    __slots__ = '_folder_info', '_frame'

    def __init__(self, folder_info: ontology_info_like, frame: str):
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

    def open_mask(self, path: path_like):
        path = self._folder_info.folder() / self._frame / (str(path) + '.png')
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

    def __init__(self, folder: image_folder_info_like):
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

    def write(self) -> 'files':
        assert not self._folder.exists()
        for k, v in self.__class__.__dict__.items():
            if k.endswith('folder') and k not in self.__slots__:
                v(self).mkdir(parents=True, exist_ok=True)
        spec_src = default_forms_folder / 'image_folder_specification.yml'
        spec_dst = self.specification_path()
        shutil.copy(spec_src, spec_dst)

    def specification_path(self) -> pathlib.Path:
        return self._folder / 'image_folder_specification.yml'

    def specification(self) -> dict:
        with self.specification_path().open('rt') as f:
            spec = yaml.safe_load(f.read())
        return spec

    def __iter__(self) -> image_info_iterator_type:
        for p in self.image_folder().iterdir():
            n = p.with_suffix('').name
            yield ImageInfo(self, n)

    def raw_iter(self) -> image_info_iterator_type:
        for p in self.raw_image_folder().iterdir():
            n = p.stem
            yield ImageInfo(self, n)

    def image_info(self, name: str):
        return ImageInfo(self, name)

    def __len__(self):
        for i, p in enumerate(self.image_folder().iterdir()):
            pass
        return i + 1


class ImageInfo:
    __slots__ = '_folder_info', '_name'

    def __init__(self, folder_info: image_info_like, name: str):
        self._folder_info = ImageFolderInfo(folder_info)
        self._name = name

    def name(self) -> str:
        return self._name

    def folder_info(self) -> 'ImageFolderInfo':
        return self._folder_info

    def raw_image_path(self) -> pathlib.Path:
        return self._folder_info.raw_image_folder() / self._name

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

    def __init__(self, image_folder_info: image_folder_info_like,
                 ontology_folder_info: ontology_folder_info_like,
                 folder: segmentation_result_folder_info_like):
        if isinstance(folder, self.__class__):
            self._image_folder_info = ImageFolderInfo(image_folder_info) if image_folder_info \
                else folder.image_folder_info()
            self._ontology_folder_info = OntologyFolderInfo(ontology_folder_info) if ontology_folder_info \
                else folder.ontology_folder_info()
            self._folder = folder.folder()
        else:
            self._image_folder_info = ImageFolderInfo(image_folder_info)
            self._ontology_folder_info = OntologyFolderInfo(ontology_folder_info)
            self._folder = pathlib.Path(folder)

    def image_folder_info(self):
        return self._image_folder_info

    def ontology_folder_info(self):
        return self._ontology_folder_info

    def folder(self):
        return self._folder

    def specification_folder(self):
        return self._folder / 'spec'

    def specification_path(self) -> pathlib.Path:
        return self.specification_folder() / 'specification_form.yml'

    def specification(self):
        with self.specification_path().open('rt') as f:
            spec = yaml.safe_load(f.read())
        return spec

    def structure_list_path(self):
        return self.specification_folder() / 'structures.txt'

    def batches_path(self):
        return self.specification_folder() / 'batches.txt'

    def mask_permutation_path(self) -> pathlib.Path:
        return self.specification_folder() / 'mask_permutation.py'


    def segmentation_temp(self):
        return self.folder() / 'temp'

    def table_folder(self):
        return self.folder() / 'tables'

    def plot_folder(self):
        return self.folder() / 'plots'

    @classmethod
    def read(cls, folder: path_like) -> 'SegmentationResultFolderInfo':
        self = cls('', '', folder)
        spec = self.specification()['general']
        return cls(spec['image_folder'], spec['ontology_folder'], folder)

    def write(self):
        assert not self.folder().exists()
        for k, v in self.__class__.__dict__.items():
            if k.endswith('folder') and k not in self.__slots__:
                v(self).mkdir(parents=True, exist_ok=True)
        src = default_forms_folder / 'result_folder_specification.yml'
        dst = self.specification_path()
        with src.open('rt') as f:
            text = f.read()
        x = str(self._ontology_folder_info.folder().as_posix())
        text = text.replace('replace with ontology folder absolute path', x, 1)
        x = str(self._image_folder_info.folder().as_posix())
        text = text.replace('replace with image folder absolute path', x, 1)
        with dst.open('wt') as f:
            f.write(text)
