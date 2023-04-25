import shutil as sh

import numpy as np

from ..prelude import *
from .. import protocols as proto
from ..lib import filesystem as fs, io_utils as io, config_utils

# <editor-fold desc="InfoI implementations">


def _create_v2_compat_info(obj: proto.InfoI):
    """preferred way to create data directories"""
    assert not obj.path().exists()
    for k, v in obj.__class__.__dict__.items():
        if k.endswith("dir") and hasattr(v, "__call__"):
            v(obj).mkdir(parents=True, exist_ok=True)
    spec_src = obj.default_config()
    spec_dst = obj.config()
    sh.copy(spec_src, spec_dst)
    obj.tag_file().write(obj.tag())


@proto.implements(proto.InfoI)
@proto.implements(proto.ImageInfoI)
class ImageDirInfo:
    __slots__ = ("_path",)

    def __init__(self, path):
        self._path = path

    TAG = "BrainS.v2compat.ImageDirInfo"

    # default implementations
    TAG_FILE_NAME = proto.InfoI.TAG_FILE_NAME
    __repr__ = repr_slots
    tag = proto.InfoI.tag
    tag_file = proto.InfoI.tag_file
    create = _create_v2_compat_info

    # InfoI impl
    def path(self) -> Path:
        return self._path

    def __fspath__(self) -> str:
        return str(self._path)

    def config(self) -> fs.File:
        return io.YamlFile(self._path / "configuration")

    @staticmethod
    def default_config() -> fs.File:
        return io.YamlFile(
            cfg.resource_dir / "v2/image_folder_configuration.yml"
        )

    @classmethod
    def read_from(
        cls, directory: Path, *, fix_tag=True, check_naming=False
    ) -> "ImageDirInfo":
        assert directory.exists()
        self = cls(directory)
        assert Path(self.config().path).exists()
        tag_f = self.tag_file()
        if not Path(tag_f.path).exists() and fix_tag:
            tag_f.write(self.tag())
        if check_naming:
            self.check_naming()
        return self

    # ImageInfoI & related
    def meta_dir(self) -> Path:
        return self._path / "meta"

    def pre_meta_dir(self) -> Path:
        return self._path / "pre_meta"

    def raw_image_dir(self) -> Path:
        return self._path / "img_raw"

    def image_dir(self) -> Path:
        return self._path / "img"

    def cropped_image_dir(self) -> Path:
        return self._path / "img_cropped"

    def raw_images(
        self, dtype="<i4", shape=(256, 256)
    ) -> fs.FileTable[str, io.np.ndarray]:
        def read_raw(path) -> io.np.ndarray:
            return io.np.fromfile(path, dtype=dtype).reshape(shape)

        return fs.PrefixSuffixFormatter(
            self.raw_image_dir(), suffix=""
        ).to_FileTable(read_raw)

    def images(self) -> fs.FileTable[str, io.np.ndarray]:
        return io.NpyDir(self.image_dir())

    def metas(self) -> fs.FileTable[str, io.Yaml]:
        return io.YamlDir(self.meta_dir())

    def cropped_images(self) -> fs.FileTable[str, io.np.ndarray]:
        return io.NpyDir(self.cropped_image_dir())

    __DS_METHS = (
        (images, image_dir),
        (metas, meta_dir),
        (cropped_images, cropped_image_dir),
        (raw_images, raw_image_dir),
        (None, pre_meta_dir),
    )

    def check_naming(self):
        """
        :return: check non-empty directories for naming consistency.
        if inconsistent, `raise Err(non_existent_keys)`
        """
        proto_tables = [
            (di(self).name, ft)
            for (ft, di) in self.__DS_METHS
            if di(self).exists()
            and not (ft is None or fs.is_empty_dir(di(self)))
        ]
        if len(proto_tables) < 2:
            return  # nothing to check
        tables: List[Tuple[str, fs.FileTable]] = [
            (name, ft(self)) for name, ft in proto_tables
        ]
        head = tables[0][1]
        non_existent = []
        for key in head.keys():
            non_existent.extend(
                (name, key)
                for name, ft in tables
                if not Path(ft.format(key)).exists()
            )
        if non_existent:
            raise Err(non_existent)


# @proto.implements(proto.InfoI)
# @proto.implements(proto.AtlasInfoI)
class AtlasDirInfo:
    ...


# @proto.implements(proto.InfoI)
# @proto.implements(proto.SegmentationInfoI)
class SegmentationDirInfo:
    ...


# </editor-fold>


# <editor-fold description="pipeline implementations">


# <editor-fold description="collect metadata">


def collect_image_metadata(image_dir: os.PathLike) -> List[Err]:
    image_dir = ImageDirInfo.read_from(Path(image_dir))
    c = image_dir.config().read()
    read_pre_meta = __pre_meta_reader(
        image_dir.pre_meta_dir(), c["metadata_keys"], c["file_name_fields"]
    )
    metas = image_dir.metas()
    keys = image_dir.raw_images().keys()
    # del c, image_dir

    errors = []
    for k in keys:
        d = read_pre_meta(k)
        if type(d) != dict:
            errors.append(d)
        else:
            metas[k] = d
    return errors


def __pre_meta_reader(
    prefix: Path, metadata_keys: dict, field_names: List[str]
) -> Fn[[str], Union[dict, Err]]:
    config_list = [
        # key, dtype, indexing array
        (k, np.dtype(v["dtype"]), np.array(v["indices"], dtype=int))
        for k, v in metadata_keys.items()
    ]
    del metadata_keys

    def clj(k: str) -> Union[dict, Err]:
        d = {k: v for k, v in zip(field_names, k.split("_"))}
        for meta_key, dt, idx in config_list:
            p = prefix / f"{k}_{meta_key}.txt"
            if not p.exists():
                return Err(
                    (k, meta_key, FileExistsError, p)
                )  # I've warned, it would be very unpythonic! )
            try:
                chunck = np.fromfile(p, sep="\t")[idx].astype(dt).tolist()
                d[k] = chunck if len(chunck) > 1 else chunck[0]
            except Exception as e:
                return Err((k, meta_key, type(e), e.args))
        return d

    return clj


# </editor-fold>


def raw_images_to_npy(image_dir: os.PathLike, dtype="<i4", shape=(256, 256)):
    image_dir = ImageDirInfo.read_from(Path(image_dir))
    raw = image_dir.raw_images(dtype, shape)
    imgs = image_dir.images()
    errors = []
    for k in raw.keys():
        try:
            imgs[k] = raw[k]
        except Exception as e:
            errors.append(Err((k, type(e), e.args)))
    return errors


# <editor-fold description="crop images">
def crop_images(img_dir: ImageDirInfo):
    ...


# </editor-fold>


# <editor-fold description="average cropped images">
def average_cropped_images():
    ...


# </editor-fold>

# <editor-fold description="download svgs">


class atlas_download:
    @staticmethod
    def slice_ids_table():
        ...

    @staticmethod
    def default_ontology():
        ...

    @staticmethod
    def svgs():
        ...


# </editor-fold>

# <editor-fold description="prerender masks">


def prerender_masks(atlas_dir: AtlasDirInfo):
    ...


# </editor-fold>


# <editor-fold description="">

# </editor-fold>
# <editor-fold description="">
# </editor-fold>
# <editor-fold description="">
# </editor-fold>
# <editor-fold description="average cropped images">
# </editor-fold>

# </editor-fold>
