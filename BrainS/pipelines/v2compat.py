import shutil as sh
from argparse import Namespace as NS
import xml.etree.ElementTree as et

import numpy as np
import pandas as pd

from ..lib.functional import ValDispatch
from ..prelude import *
from .. import protocols as proto
from ..lib import filesystem as fs, io_utils as io, config_utils, iterators

__all__ = [
    "ImageDirInfo",
    "AtlasDirInfo",
    "SegmentationDirInfo",
    "PipelineErrors",
    "_new_err",
    "explain_error",
    "ListErrorLog",
    "collect_image_metadata",
    "atlas_download",
    "compress_image_dir",
]
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

    def __init__(self, path: Path):
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
        return io.YamlFile(self._path / "image_folder_configuration.yml")

    @staticmethod
    def default_config() -> fs.File:
        return io.YamlFile(
            cfg.resource_dir / "v2/image_folder_configuration.yml"
        )

    @classmethod
    def read_from(
        cls, directory: os.PathLike, *, fix_tag=True, check_naming=False
    ) -> "ImageDirInfo":
        """
        :param directory: if directory is an ImageDirInfo, returns a shallow copy without any validation
        :param fix_tag:
        :param check_naming:
        :return:
        """
        if isa(directory, cls):
            return cls(directory._path)  # type: ignore
        directory = Path(directory)
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
    def metadata_dir(self) -> Path:
        return self._path / "meta"

    def pre_metadata_dir(self) -> Path:
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

    def metadata(self) -> fs.FileTable[str, io.Yaml]:
        return io.YamlDir(self.metadata_dir())

    def cropped_images(self) -> fs.FileTable[str, io.np.ndarray]:
        return io.NpyDir(self.cropped_image_dir())

    __DS_METHS = (
        (images, image_dir),
        (metadata, metadata_dir),
        (cropped_images, cropped_image_dir),
        (raw_images, raw_image_dir),
        (None, pre_metadata_dir),
    )

    def check_naming(self):
        """
        :return: check non-empty directories for naming consistency.
        if inconsistent, then
        :raise `` Err((PipelineErrors.KEYS_NOT_EXIST, non_existent_keys))``
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
            raise Err((PipelineErrors.KEYS_NOT_EXIST, non_existent))


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

# <editor-fold descr="prerequisites">

"""
this section may seem overly dynamic. usually, type fluency is a bad thing.
however, data-processing pipelines are very susceptible to changes and corrections,
so argument broadcasting, non-specific type definitions,
mutable dispatches and enums (argparse.Namespace) can be a good fit here.
they make things easier to hack from outside:
you do not need to change this module's code, to change the behaviour.
"""

PipelineErrors = _E = NS()


def _new_err(name: str):
    """
    register a new PipelineError.
    this allows to distribute the error definitions,
    yet collect all errors in a single object
    """
    setattr(PipelineErrors, name, name)


@ValDispatch.new(None)
def explain_error(*_) -> str:
    """explain one of the pipeline errors arguments"""
    raise NotImplementedError


class ListErrorLog:
    __slots__ = "lst", "stream"

    def __init__(self, stream: Opt[Any] = sys.stderr):
        self.lst = []
        self.stream = stream

    def __call__(self, e: Opt[Err]):
        if not is_err(e):
            return
        self.lst.append(e)
        if self.stream is not None:
            self.stream.write(explain_error(*e.data))


_log_t = Fn[[Opt[Err]], None]


# </editor-fold>


# <editor-fold descr="zip ImageDir">
def compress_image_dir(image_dir, zip_file: os.PathLike = None, use_raw=False):
    """
    :param image_dir: ``ImageInfoI`` implementation or path
    :param zip_file: path to a .zip file
    :param use_raw: assume path is ImageDir
    """
    # TODO: proper error handling
    imd = proto.coerce_to(proto.ImageInfoI, image_dir, ImageDirInfo.read_from)
    if use_raw:
        dirs = imd.raw_image_dir(), imd.pre_metadata_dir()  # type: ignore  # not ImageInfo-compliant
    else:
        dirs = imd.image_dir(), imd.metadata_dir()

    assert all(d.exists() for d in dirs), f"some of {dirs} do not exist"
    files = iterators.chain(
        (imd.config().path, imd.tag_file().path),
        *(d.iterdir() for d in dirs)
    )
    src = imd.path()
    zip_file = zip_file or (str(src) + ".backup.zip")
    io.zip_files(zip_file, files, src)

# </editor-fold>


# <editor-fold description="collect metadata">

_new_err("PRE_METADATA_NOT_FOUND")
_new_err("PRE_METADATA_READ_ERROR")


_pre_meta_reader_t = Fn[[Path, str], Union[dict, Err]]


def _pre_meta_reader(
    metadata_keys: dict, field_names: List[str]
) -> _pre_meta_reader_t:
    config_list = [
        # key, dtype, indexing array
        (k, np.dtype(v["dtype"]), np.array(v["indices"], dtype=int))
        for k, v in metadata_keys.items()
    ]
    del metadata_keys

    def clj(prefix: Path, k: str) -> Union[dict, Err]:
        d = {k: v for k, v in zip(field_names, k.split("_"))}
        for meta_key, dt, idx in config_list:
            p = prefix / f"{k}_{meta_key}.txt"
            if not p.exists():
                return Err(
                    (_E.PRE_METADATA_NOT_FOUND, k, meta_key, p)
                )  # I've warned, it would be very unpythonic! )
            try:
                chunck = np.fromfile(p, sep="\t")[idx].astype(dt).tolist()
                d[k] = chunck if len(chunck) > 1 else chunck[0]
            except Exception as e:
                return Err(
                    (
                        _E.PRE_METADATA_READ_ERROR,
                        k,
                        meta_key,
                        type(e),
                        e.args,
                        e.__traceback__.tb_frame.f_lineno,
                    )
                )
        return d

    return clj


@explain_error.register(_E.PRE_METADATA_NOT_FOUND)
def explain_error(_, k, meta_key, path):
    return f"pipeline: {collect_image_metadata.__name__}, step: {(k, meta_key)}, cause: missing {path}"


@explain_error.register(_E.PRE_METADATA_READ_ERROR)
def explain_error(_, k, meta_key, exc_type, exc_args, lineno):
    return (
        f"pipeline: {collect_image_metadata.__name__}, step: {(k, meta_key)}, "
        f"line: {__name__}::{lineno} cause: metadata preprocessing error: {exc_type}{exc_args}"
    )


class collect_image_metadata(NamedTuple):
    pre_meta_reader: _pre_meta_reader_t

    @classmethod
    def _from_dict_interface_(cls, d: dict) -> "collect_image_metadata":
        r = _pre_meta_reader(d["metadata_keys"], d["file_name_fields"])
        return cls(r)

    @classmethod
    def get_main_loop(
        cls, image_dir: os.PathLike
    ) -> Tuple[Fn[[str], Opt[Err]], Iterable[str]]:
        image_dir = ImageDirInfo.read_from(image_dir)
        image_dir_config = image_dir.config().read()
        self = cls._from_dict_interface_(image_dir_config)

        metas = image_dir.metadata()
        keys = image_dir.keys()
        pre_meta_dir = image_dir.pre_metadata_dir()
        read_pre_meta = self.pre_meta_reader

        def main_loop(k: str) -> Opt[Err]:
            d = read_pre_meta(pre_meta_dir, k)
            return d if is_err(d) else metas.__setitem__(k, d)

        return main_loop, keys

    @classmethod
    def main(cls, image_dir: os.PathLike, log: _log_t = None) -> _log_t:
        loop, keys = cls.get_main_loop(image_dir)
        log = ListErrorLog() if log is None else log
        for k in keys:
            log(loop(k))
        return log


# </editor-fold>


# <editor-fold descr="raw images to npy">
_new_err("RAW_IMAGES_TO_NUMPY")


@explain_error.register(_E.RAW_IMAGES_TO_NUMPY)
def explain_error(_, k, exc_type, exc_args) -> str:
    return f"pipeline: {raw_images_to_npy.__name__}, step: {k}, cause: exception {exc_type}{exc_args}"


def raw_images_to_npy(image_dir: os.PathLike, dtype="<i4", shape=(256, 256), log: _log_t = None) -> _log_t:
    image_dir = ImageDirInfo.read_from(image_dir)
    raw = image_dir.raw_images(dtype, shape)
    imgs = image_dir.images()
    log = ListErrorLog() if log is None else log
    for k in raw.keys():
        try:
            imgs[k] = raw[k]
        except Exception as e:
            log(Err((_E.RAW_IMAGES_TO_NUMPY, k, type(e), e.args)))
    return log
# </editor-fold>


# <editor-fold description="crop images">
def crop_images(img_dir: os.PathLike):
    ...


# </editor-fold>


# <editor-fold description="average cropped images">
def average_cropped_images():
    ...


# </editor-fold>


# <editor-fold description="download svgs">


class atlas_download:
    @staticmethod
    def slice_ids_table(atlas_dir: os.PathLike) -> Tuple[Path, pd.DataFrame]:
        from urllib.request import urlretrieve as download
        atlas_dir = proto.coerce_to(proto.AtlasInfoI, atlas_dir, AtlasDirInfo.read_from)
        save_path = atlas_dir.path() / "slice_ids.txt"
        config = atlas_dir.config().read()
        # perhaps, move to _from_dict_interface_. or maybe not
        kwargs = config["downloading_arguments"]
        url = config["downloading_urls"]["slice_ids"]
        url = url.format(**kwargs)
        download(url, save_path)
        root = et.parse(save_path).getroot()
        slice_ids = [int(node.text) for node in root.iter(tag="id")]
        del root
        slice_ids.reverse()
        # pair with coordinates
        slice_ids = np.array(slice_ids)
        coords = np.linspace(**kwargs["slice_coordinates"], num=len(slice_ids))
        t = pd.DataFrame.from_dict({"ids": slice_ids, "coordinates": coords})
        t.to_csv(save_path, sep="\t", index=False)
        return save_path, t

    @staticmethod
    def default_ontology(atlas_dir: os.PathLike):
        from urllib.request import urlretrieve as download
        atlas_dir = proto.coerce_to(proto.AtlasInfoI, atlas_dir, AtlasDirInfo.read_from)
        ...

    @staticmethod
    def svgs(atlas_dir: os.PathLike):
        from urllib.request import urlretrieve as download
        atlas_dir = proto.coerce_to(proto.AtlasInfoI, atlas_dir, AtlasDirInfo.read_from)
        ...


# </editor-fold>


# <editor-fold description="prerender masks">


def prerender_masks(atlas_dir: os.PathLike):
    atlas_dir = AtlasDirInfo.read_from(atlas_dir)


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
