"""
This module contains utility classes that represent abstract concepts
like research projects, atlas sections, etc. All interfacing with
these constructs (reading data, writing) should be done through the classes declared here.

All the instances of these classes are lightweight (they contain 1-3 fields),
they nave very little initialization logic. However, large portion of their methods performs IO operations.
"""


import yaml
import shutil
from pathlib import Path
import numpy as np
import typing
from xml.etree import ElementTree
import PIL
from copy import deepcopy

from BrainS.prelude import cfg as _future_cfg

# constants
# _DEFAULT_FOLDER_CONFIGURATIONS_STORAGE = (
#     Path(__file__).parent.parent / "folder_configurations_default"
# ) # paths to default configs are now configured by `_future_cfg`

_ONTOLOGY_FOLDER_CONFIGURATION_NAME = "ontology_folder_configuration.yml"
_IMAGE_FOLDER_CONFIGURATION_NAME = "image_folder_configuration.yml"
_RESULTS_FOLDER_CONFIGURATION_NAME = "results_folder_configuration.yml"
_STRING_REPLACED_WITH_IMAGE_FOLDER = "replace with image folder absolute path"
_STRING_REPLACED_WITH_ONTOLOGY_FOLDER = (
    "replace with ontology folder absolute path"
)

# type aliases
path_like = typing.Union[Path, str]
ontology_folder_info_like = typing.Union["OntologyFolderInfo", path_like]
ontology_info_like = typing.Union["OntologyInfo", path_like]
image_folder_info_like = typing.Union["ImageFolderInfo", path_like]
image_info_like = typing.Union["ImageInfo", path_like]
segmentation_result_folder_info_like = typing.Union[
    "SegmentationResultFolderInfo", path_like
]
image_info_iterator_type = typing.Iterator["ImageInfo"]
ontology_info_iterator_type = typing.Iterator["OntologyInfo"]


class OntologyFolderInfo:
    """
    Instances of this class interface with the filesystem
    to provide a simple way of managing folder with atlas sections.
    """

    __slots__ = ("_folder",)

    def __init__(self, folder: ontology_folder_info_like):
        """
        Creates an instance from a `Path` or string representation of a `Path`.
        If `OntologyFolderInfo` instance is provided as a `folder`,
        copies data from the instance (this is done for utility purposes).
        """
        self._folder = (
            deepcopy(folder._folder)
            if isinstance(folder, self.__class__)
            else Path(folder)
        )

    def __repr__(self) -> str:
        return f"{self.__class__}({self._folder.absolute().as_posix()})"

    def folder(self) -> Path:
        """:returns root folder"""
        return self._folder

    def onts_folder(self) -> Path:
        """:returns folder with `.xml` ontologies"""
        return self._folder / "onts"

    def svgs_folder(self) -> Path:
        """:returns folder with atlas `.svg` sections"""
        return self._folder / "svgs"

    def ontology_info(self, frame: str) -> "OntologyInfo":
        """shorthand for `OntologyInfo` initialization"""
        return OntologyInfo(self, frame)

    def configuration_path(self) -> Path:
        return self._folder / _ONTOLOGY_FOLDER_CONFIGURATION_NAME

    def configuration(self) -> dict:
        """reads and :returns `.yml` config file"""
        with self.configuration_path().open("rt") as f:
            s = yaml.safe_load(f.read())
        return s

    def write(self) -> None:
        """A preferred way to create folder and files,
        :raises `FileExistsError` if the folder already exists"""
        if self._folder.exists():
            raise FileExistsError(f"{self._folder} should not exist.")
        # self._folder.mkdir(exist_ok=False, parents=True)
        for k, v in self.__class__.__dict__.items():
            if k.endswith("folder") and k not in self.__slots__:
                v(self).mkdir(parents=True, exist_ok=True)
        form_src = _future_cfg.resource_dir / "v2" / _ONTOLOGY_FOLDER_CONFIGURATION_NAME  # type: ignore
        form_dst = self.configuration_path()
        shutil.copy(form_src, form_dst)

    def frames(self) -> typing.List[str]:
        """:returns all the file names from the `self.svgs_folder()`"""
        return [
            p.stem for p in self.svgs_folder().iterdir() if p.suffix == ".svg"
        ]

    def __iter__(self) -> ontology_info_iterator_type:
        """yields OntologyInfo for each frame in `self.frames()`"""
        for f in self.frames():
            yield OntologyInfo(self, f)

    def items(self) -> typing.Iterator[typing.Tuple[str, "OntologyInfo"]]:
        """Analog of `dict.items()`: makes an iterator that
        yields `(frame, OntologyInfo)` pairs
        """
        for f in self.frames():
            yield f, OntologyInfo(self, f)


class OntologyInfo:
    """
    Instances of this class interface with the filesystem
    to provide a simple way of managing individual atlas sections.
    """

    __slots__ = "_folder_info", "_frame"

    def __init__(self, folder_info: ontology_folder_info_like, frame: str):
        self._folder_info = OntologyFolderInfo(folder_info)
        self._frame = str(frame)

    def __repr__(self) -> str:
        return f"{self.__class__}({self._folder_info.folder().absolute().as_posix()}, {self._frame})"

    def folder_info(self) -> OntologyFolderInfo:
        """:returns `OntologyFolderInfo` instance (no copying is done)"""
        return self._folder_info

    def frame(self) -> str:
        """:returns frame (section) name"""
        return self._frame

    def get_mask_filename(
        self, mask_name: str = "Root"
    ) -> typing.Union[Path, None]:
        """
        Performs linear search in ontology tree.
        :var mask_name must match exactly
        :returns Path if structure is found and has 'filename' attribute, else None
        """
        rt = self.tree().getroot()
        for s in rt.iter("structure"):
            if s.attrib["name"] == mask_name:
                fn = s.attrib["filename"]
                return Path(fn) if fn else None

    def masks_folder(self) -> Path:
        """:returns folder where the masks are stored"""
        return self._folder_info.folder() / self._frame

    def mask_path_absolute(
        self, mask_name: str = "Root"
    ) -> typing.Optional[Path]:
        """
        Performs linear search in ontology tree.
        :var mask_name must match exactly
        :returns Path if structure is found and has 'filename' attribute, else None
        """
        p = self.get_mask_filename(mask_name)
        return self.masks_folder() / p.with_suffix(".png")

    @staticmethod
    def open_mask(path: path_like) -> "np.ndarray[bool]":
        """
        :returns boolean mask of an anatomical structure
        this method does not modify the file path
        """
        mask = PIL.Image.open(path)
        mask = np.array(mask.getdata(), dtype=np.uint8).reshape(
            (mask.size[1], mask.size[0])
        )
        return mask > 127

    def open_mask_relative(self, path: path_like) -> "np.ndarray[bool]":
        """
        :param path normally is taken from attributes of `self.tree()` nodes.
        therefore file suffix is changed to `.png` and path is considered
        relative to `self.masks_folder()`
        """
        path = (self.masks_folder() / path).with_suffix(".png")
        return self.open_mask(path)

    def tree_path(self) -> Path:
        """:returns a path to the `.xml` ontology of the section"""
        return self._folder_info.onts_folder() / (self._frame + ".xml")

    def tree(self) -> ElementTree.ElementTree:
        """reads and :returns ontology of the section"""
        return ElementTree.parse(self.tree_path())

    def svg_path(self) -> Path:
        """:returns path to the `.svg` atlas section"""
        return self._folder_info.svgs_folder() / (self._frame + ".svg")

    def svg(self) -> ElementTree.ElementTree:
        """reads and :returns content of `.svg` section"""
        return ElementTree.parse(self.svg_path())

    def default_tree_path(self) -> Path:
        """:returns path to a default `.xml` section ontology
        (downloaded from Allen Institute)"""
        return self._folder_info.onts_folder() / "default.xml"

    def default_tree(self) -> ElementTree.ElementTree:
        """reads and :returns content of default `.xml` ontology"""
        return ElementTree.parse(self.default_tree_path())


class ImageFolderInfo:
    """This class corresponds to the folder where
    the source images and their metadata are stored"""

    __slots__ = "_folder"

    def __init__(self, folder: image_folder_info_like):
        """
        Creates an instance from a `Path` or string representation of a `Path`.
        If `ImageFolderInfo` instance is provided as a `folder`,
        copies data from the instance (this is done for utility purposes).
        """
        self._folder = (
            deepcopy(folder._folder)
            if isinstance(folder, self.__class__)
            else Path(folder)
        )

    def __repr__(self) -> str:
        return f"{self.__class__}({self._folder.absolute().as_posix()})"

    def folder(self) -> Path:
        """:returns the folder passed on initialization"""
        return self._folder

    def image_folder(self) -> Path:
        """:returns folder where uncropped `.npy` images are stored"""
        return self._folder / "img"

    def metadata_folder(self) -> Path:
        """:returns folder that stores image metadata"""
        return self._folder / "meta"

    def cropped_image_folder(self) -> Path:
        """:returns folder with cropped `.npy` images"""
        return self._folder / "img_cropped"

    def pre_metadata_folder(self) -> Path:
        """:returns folder that stores simple text files (later turned to metadata)"""
        return self._folder / "pre_meta"

    def raw_image_folder(self) -> Path:
        return self._folder / "img_raw"

    def configuration_path(self) -> Path:
        """:returns path to the image folder config"""
        return self._folder / _IMAGE_FOLDER_CONFIGURATION_NAME

    def configuration(self) -> dict:
        """reads and returns this image folder config"""
        with self.configuration_path().open("rt") as f:
            spec = yaml.safe_load(f.read())
        return spec

    def write(self) -> None:
        """preferred way of writing the folder that this class represents"""
        assert not self._folder.exists()
        for k, v in self.__class__.__dict__.items():
            if k.endswith("folder") and k not in self.__slots__:
                v(self).mkdir(parents=True, exist_ok=True)
        spec_src = _future_cfg.resource_dir / "v2" / _IMAGE_FOLDER_CONFIGURATION_NAME  # type: ignore
        spec_dst = self.configuration_path()
        shutil.copy(spec_src, spec_dst)

    def image_info(self, name: str) -> "ImageInfo":
        """shorthand `ImageInfo` constructor"""
        return ImageInfo(self, name)

    def __iter__(self) -> image_info_iterator_type:
        """
        yields `ImageInfo` instances
        !NB: uses `self.image_folder()` to deduce image names
        """
        for p in self.image_folder().iterdir():
            n = p.with_suffix("").name
            yield ImageInfo(self, n)

    def raw_iter(self) -> image_info_iterator_type:
        """
        scans `self.raw_image_folder()`, yields `ImageInfo` instances
        """
        for p in self.raw_image_folder().iterdir():
            n = p.stem
            yield ImageInfo(self, n)

    def __len__(self):
        """returns number of images in `self.image_folder()`"""
        for i, p in enumerate(self.image_folder().iterdir()):
            pass
        return i + 1


class ImageInfo:
    """
    Serves for retrieving different representations of a single tomographic section,
    as well as it's metadata
    """

    __slots__ = "_folder_info", "_name"

    def __init__(self, folder_info: image_folder_info_like, name: str):
        self._folder_info = ImageFolderInfo(folder_info)
        self._name = name

    def __repr__(self) -> str:
        return f"{self.__class__}({self._folder_info.folder().absolute().as_posix()}, {self._name})"

    def name(self) -> str:
        """:returns name of the image"""
        return self._name

    def folder_info(self) -> ImageFolderInfo:
        """:returns name of"""
        return self._folder_info

    def raw_image_path(self) -> Path:
        """:returns path of raw image version"""
        return (
            self._folder_info.raw_image_folder() / self._name
        )  # (self._name + ".img")

    def raw_image(self, dtype="<i4", shape=(256, 256)) -> np.ndarray:
        """reads and :returns raw image as ndarray"""
        return np.fromfile(self.raw_image_path(), dtype=dtype).reshape(shape)

    def image_path(self) -> Path:
        """:returns path to uncropped `.npy` image"""
        return self._folder_info.image_folder() / (self._name + ".npy")

    def image(self) -> np.ndarray:
        """reads and :returns uncropped `.npy` image"""
        return np.load(self.image_path())

    def cropped_image_path(self) -> Path:
        """:returns path to cropped `.npy` image"""
        return self._folder_info.cropped_image_folder() / (self._name + ".npy")

    def cropped_image(self) -> np.ndarray:
        """reads and :returns cropped `.npy` image"""
        return np.load(self.cropped_image_path())

    def metadata_path(self) -> Path:
        """:returns path to `.yml` metadata of this image"""
        return self._folder_info.metadata_folder() / (self._name + ".yml")

    def metadata(self) -> dict:
        """reads and :returns `.yml` metadata of this image"""
        with self.metadata_path().open("rt") as f:
            m = yaml.safe_load(f.read())
        return m


class SegmentationResultFolderInfo:
    """
    For managing folder with segmentation results. Normally it is initialized from existing folder.
    """

    __slots__ = "_image_folder_info", "_ontology_folder_info", "_folder"

    def __init__(
        self,
        image_folder_info: image_folder_info_like,
        ontology_folder_info: ontology_folder_info_like,
        folder: segmentation_result_folder_info_like,
    ):
        """
        This is a first-time initialization method
        For initializing instance that corresponds to existing folder
        `self.read()` initialization method is preferable (less error-prone).

        If `SegmentationResultFolderInfo` instance is provided as a `folder`,
        copies `folder` from the instance (this is done for utility purposes).
        """
        self._image_folder_info = ImageFolderInfo(image_folder_info)
        self._ontology_folder_info = OntologyFolderInfo(ontology_folder_info)
        self._folder = (
            deepcopy(folder.folder())
            if isinstance(folder, self.__class__)
            else Path(folder)
        )

    def image_folder_info(self) -> ImageFolderInfo:
        """:returns `ImageFolderInfo` associated with the instance"""
        return self._image_folder_info

    def ontology_folder_info(self) -> OntologyFolderInfo:
        """:returns `OntologyFolderInfo` associated with this instance"""
        return self._ontology_folder_info

    def folder(self) -> Path:
        """:returns root folder"""
        return self._folder

    def configuration_folder(self) -> Path:
        """:returns folder with configs"""
        return self._folder / "spec"

    def configuration_path(self) -> Path:
        """:returns path to the folder config"""
        return self.configuration_folder() / _RESULTS_FOLDER_CONFIGURATION_NAME

    def configuration(self) -> dict:
        """reads and :returns folder config"""
        with self.configuration_path().open("rt") as f:
            spec = yaml.safe_load(f.read())
        return spec

    def structure_list_path(self) -> Path:
        """:returns path to the list of structures involved into segmentation"""
        return self.configuration_folder() / "structures.txt"

    def batches_path(self) -> Path:
        """:returns path to the file that specifies groups, etc during segmentation."""
        return self.configuration_folder() / "batches.txt"

    def mask_permutation_path(self) -> Path:
        """:returns path to the file that specifies how each structure mask is perturbed"""
        return self.configuration_folder() / "mask_permutation.py"

    def segmentation_temp(self) -> Path:
        """:returns path to the folder that stores intermediate segmentation files"""
        return self.folder() / "temp"

    def table_folder(self) -> Path:
        """:returns folder that stores output tables"""
        return self.folder() / "tables"

    def plot_folder(self) -> Path:
        """:returns folder that stores output plots"""
        return self.folder() / "plots"

    @classmethod
    def read(cls, folder: path_like) -> "SegmentationResultFolderInfo":
        """
        A much less error-prone constructor, that reads initialization parameters from
        the segmentation result folder configuration file.
        Additionally checks if image and ontology folders do exist.
        :param folder: path to the segmentation results folder.
        :raises FileNotFoundError if one of the folders does not exist
        """
        dummy_instance = cls("", "", folder)
        spec = dummy_instance.configuration()["general"]
        im_f, ont_f = Path(spec["image_folder"]), Path(spec["ontology_folder"])

        if not (im_f.exists() and ont_f.exists()):
            raise FileNotFoundError(f"{im_f} and {ont_f} do not exist!")
        if not im_f.exists():
            raise FileNotFoundError(f"{im_f} does not exist!")
        if not ont_f.exists():
            raise FileNotFoundError(f"{ont_f} does not exist!")

        return cls(im_f, ont_f, folder)

    def write(self) -> None:
        """For creating new segmentation results folder in the filesystem"""
        if self._folder.exists():
            raise FileExistsError(f"{self._folder} should not exist")
        for k, v in self.__class__.__dict__.items():
            if k.endswith("folder") and k not in self.__slots__:
                v(self).mkdir(parents=True, exist_ok=True)
        src = _future_cfg.resource_dir / "v2" / _RESULTS_FOLDER_CONFIGURATION_NAME  # type: ignore
        dst = self.configuration_path()
        with src.open("rt") as f:
            text = f.read()
        x = str(self._ontology_folder_info.folder().as_posix())
        text = text.replace(_STRING_REPLACED_WITH_ONTOLOGY_FOLDER, x, 1)
        x = str(self._image_folder_info.folder().as_posix())
        text = text.replace(_STRING_REPLACED_WITH_IMAGE_FOLDER, x, 1)
        with dst.open("wt") as f:
            f.write(text)
