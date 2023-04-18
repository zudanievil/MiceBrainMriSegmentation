from ..prelude import *
from .. import protocols as proto
from ..lib import filesystem as fs


@proto.implements(proto.InfoI)
@proto.implements(proto.ImageInfoI)
class ImageDirInfo:  # implements
    __slots__ = ("_path",)

    def path(self) -> Path:
        return self._path

    def __fspath__(self) -> str:
        return str(self._path)

    def config(self) -> fs.File:
        ...

    def default_config(self) -> fs.File:
        ...

    def __repr__(self) -> str:
        ...

    @classmethod
    def from_path(self, folder: Path) -> "InfoI":
        ...

    def create(self) -> "InfoI":
        ...

    def images(self) -> fs.FileTable:
        ...

    def metadata(self) -> fs.FileTable:
        ...

    def cropped_images(self) -> fs.FileTable:
        ...


# @proto.impl(proto.ImageInfoI)
# class ImageFolderInfo:
#     """This class corresponds to the folder where
#     the source images and their metadata are stored"""
#     __slots__ = '_folder'
#
#     def __init__(self, folder: image_folder_info_like):
#         """
#         Creates an instance from a `Path` or string representation of a `Path`.
#         If `ImageFolderInfo` instance is provided as a `folder`,
#         copies data from the instance (this is done for utility purposes).
#         """
#         self._folder = deepcopy(folder._folder) \
#             if isinstance(folder, self.__class__) \
#             else Path(folder)
#
#     def __repr__(self) -> str:
#         return f"{self.__class__}({self._folder.absolute().as_posix()})"
#
#     def folder(self) -> Path:
#         """:returns the folder passed on initialization"""
#         return self._folder
#
#     def image_folder(self) -> Path:
#         """:returns folder where uncropped `.npy` images are stored"""
#         return self._folder / "img"
#
#     def metadata_folder(self) -> Path:
#         """:returns folder that stores image metadata"""
#         return self._folder / "meta"
#
#     def cropped_image_folder(self) -> Path:
#         """:returns folder with cropped `.npy` images"""
#         return self._folder / "img_cropped"
#
#     def pre_metadata_folder(self) -> Path:
#         """:returns folder that stores simple text files (later turned to metadata)"""
#         return self._folder / "pre_meta"
#
#     def raw_image_folder(self) -> Path:
#         return self._folder / "img_raw"
#
#     def configuration_path(self) -> Path:
#         """:returns path to the image folder config"""
#         return self._folder / _IMAGE_FOLDER_CONFIGURATION_NAME
#
#     def configuration(self) -> dict:
#         """reads and returns this image folder config"""
#         with self.configuration_path().open('rt') as f:
#             spec = yaml.safe_load(f.read())
#         return spec
#
#     def write(self) -> None:
#         """preferred way of writing the folder that this class represents"""
#         assert not self._folder.exists()
#         for k, v in self.__class__.__dict__.items():
#             if k.endswith("folder") and k not in self.__slots__:
#                 v(self).mkdir(parents=True, exist_ok=True)
#         spec_src = _DEFAULT_FOLDER_CONFIGURATIONS_STORAGE / _IMAGE_FOLDER_CONFIGURATION_NAME
#         spec_dst = self.configuration_path()
#         shutil.copy(spec_src, spec_dst)
#
#     def image_info(self, name: str) -> "ImageInfo":
#         """shorthand `ImageInfo` constructor"""
#         return ImageInfo(self, name)
