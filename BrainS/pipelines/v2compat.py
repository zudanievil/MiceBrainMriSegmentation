import shutil as sh
from argparse import Namespace as NS
import xml.etree.ElementTree as et
from typing import overload
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..lib.functional import lru_cache
from ..prelude import *
from .. import protocols as proto
from ..lib import filesystem as fs, io_utils as io, config_utils, iterators
from ._ontology import *
from ._share import *

__all__ = [
    "ImageDirInfo",
    "AtlasDirInfo",
    "SegmentationDirInfo",
    "PipelineErrors",
    "DefaultHandlers",
    "explain_error",
    "register_error",
    "ListErrorLog",
    "collect_image_metadata",
    "atlas_download",
    "compress_image_dir",
]

_E = PipelineErrors
# _H = DefaultHandlers
_soft_err_t = DefaultHandlers.soft_error_type

# <editor-fold desc="InfoI implementations">


def _create_v2_compat_info(obj: proto.InfoI, **mkdir_kw):
    """preferred way to create data directories"""
    mkdir_kw = mkdir_kw or dict(exist_ok=True, parents=True)
    assert not obj.path.exists()
    obj.path.mkdir(**mkdir_kw)
    for k, v in obj.__class__.__dict__.items():
        if k.endswith("_dir") and hasattr(v, "__call__"):
            v(obj).mkdir(**mkdir_kw)
    spec_src = obj.default_config()
    spec_dst = obj.config()
    sh.copy(spec_src, spec_dst)
    obj.tag_file().write(obj.tag())
    if isa(obj, SegmentationDirInfo):
        obj.hook_file(readonly=False).write((obj.image_info, obj.atlas_info))  # type: ignore


@proto.implements(proto.InfoI)
@proto.implements(proto.ImageInfoI)
class ImageDirInfo(NamedTuple):
    path: Path

    @classmethod
    def new(cls, path) -> "ImageDirInfo":
        return cls(Path(path))

    # InfoI impl
    def __fspath__(self) -> str:
        return str(self.path)

    TAG = "BrainS.v2compat.ImageDirInfo"

    # default implementations
    tag = proto.InfoI.tag
    tag_file = proto.InfoI.tag_file
    mkdir = _create_v2_compat_info

    def config(self) -> fs.File[dict]:
        return io.YamlFile(self.path / "image_folder_configuration.yml")

    @staticmethod
    def default_config() -> fs.File[dict]:
        return io.YamlFile(
            cfg.resource_dir / "v2/image_folder_configuration.yml"
        )

    @classmethod
    def read_from(
        cls, directory: os.PathLike, *, fix_tag=True, check_naming=False
    ) -> "ImageDirInfo":
        """
        :param directory: if directory is an ImageDirInfo, returns it without any validation
        :param fix_tag:
        :param check_naming:
        """
        if isa(directory, cls):
            return directory  # type: ignore
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
        return self.path / "meta"

    def pre_metadata_dir(self) -> Path:
        return self.path / "pre_meta"

    def raw_image_dir(self) -> Path:
        return self.path / "img_raw"

    def image_dir(self) -> Path:
        return self.path / "img"

    def cropped_image_dir(self) -> Path:
        return self.path / "img_cropped"

    def raw_images(
        self, dtype="<i4", shape=(256, 256)
    ) -> fs.FileTable[str, io.np.ndarray]:
        def read_raw(path) -> io.np.ndarray:
            return io.np.fromfile(path, dtype=dtype).reshape(shape)

        return fs.PrefixSuffixFormatter(
            self.raw_image_dir(), suffix=""
        ).to_FileTable(read_raw)

    def images(self) -> fs.FileTable[str, np.ndarray]:
        return io.NpyDir(self.image_dir())

    def metadata(self) -> fs.FileTable[str, io.Yaml]:
        return io.YamlDir(self.metadata_dir())

    def cropped_images(self) -> fs.FileTable[str, np.ndarray]:
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
        :raise ``Err((PipelineErrors.KEYS_NOT_EXIST, non_existent_keys))``
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


@proto.implements(proto.InfoI)
@proto.implements(proto.AtlasInfoI)
class AtlasDirInfo(NamedTuple):
    path: Path

    # InfoI impl
    def __fspath__(self) -> str:
        return str(self.path)

    TAG = "BrainS.v2compat.AtlasDirInfo"
    # default implementations
    tag = proto.InfoI.tag
    tag_file = proto.InfoI.tag_file
    mkdir = _create_v2_compat_info

    def config(self) -> fs.File[dict]:
        return io.YamlFile(self.path / "ontology_folder_configuration.yml")

    @staticmethod
    def default_config() -> fs.File[dict]:
        return io.YamlFile(
            cfg.resource_dir / "v2/ontology_folder_configuration.yml"
        )

    @classmethod
    def read_from(
        cls,
        directory: os.PathLike,
        *,
        fix_tag=True,
    ) -> "AtlasDirInfo":
        """
        :param directory: if directory is an AtlasDirInfo, returns it without any validation
        :param fix_tag:
        """
        if isa(directory, cls):
            return directory  # type: ignore
        directory = Path(directory)
        assert directory.exists()
        self = cls(directory)
        assert Path(self.config().path).exists()
        tag_f = self.tag_file()
        if not Path(tag_f.path).exists() and fix_tag:
            tag_f.write(self.tag())
        return self

    # AtlasInfoI impl

    def svg_dir(self) -> Path:
        return self.path / "svgs"

    def svgs(self) -> fs.FileTable:
        ...

    def ontology_dir(self) -> Path:
        return self.path / "onts"

    def ontologies(self) -> fs.FileTable[str, Ontology]:
        return fs.PrefixSuffixFormatter(
            self.ontology_dir(), ".xml"
        ).to_FileTable(read_ont, write_ont)

    def mask_dir(self) -> Path:
        return self.path

    def masks(
        self,
    ) -> fs.FileTable[str, fs.FileTable[Structure, np.ndarray[bool]]]:
        names = [p.stem for p in self.svg_dir().iterdir() if p.suffix == ".svg"]
        return fs.TableFormatter(
            {n: self.mask_dir() / n for n in names}
        ).to_FileTable(MaskDir)


_img_atlas_t = Tuple[proto.ImageInfoI, proto.AtlasInfoI]


@proto.implements(proto.InfoI)
@proto.implements(proto.SegmentationInfoI)
class SegmentationDirInfo(NamedTuple):
    path: Path
    image_info: proto.ImageInfoI
    atlas_info: proto.AtlasInfoI

    def __fspath__(self) -> str:
        return str(self.path)

    # InfoI impl
    TAG = "BrainS.v2compat.SegmentationDirInfo"

    # default implementations
    tag = proto.InfoI.tag
    tag_file = proto.InfoI.tag_file
    mkdir = _create_v2_compat_info

    def config(self) -> fs.File[dict]:
        return io.YamlFile(self.path / "results_folder_configuration.yml")

    @staticmethod
    def default_config() -> fs.File[dict]:
        return io.YamlFile(
            cfg.resource_dir / "v2/results_folder_configuration.yml"
        )

    @classmethod
    def read_from(
        cls,
        directory: os.PathLike,
        *,
        fix_tag=True,
    ) -> "SegmentationDirInfo":
        """
        :param directory: if directory is an AtlasDirInfo, returns it without any validation
        :param fix_tag:
        """
        if isa(directory, cls):
            return directory  # type: ignore
        directory = Path(directory)
        assert directory.exists()
        self = cls(directory, None, None)  # type: ignore
        image_info, atlas_info = self.hook_file().read()
        assert image_info.path.exists()
        assert atlas_info.path.exists()
        self = cls(directory, image_info, atlas_info)
        assert Path(self.config().path).exists()
        tag_f = self.tag_file()
        if not Path(tag_f.path).exists() and fix_tag:
            tag_f.write(self.tag())
        return self

    @classmethod
    def new(
        cls, image_dir, atlas_dir, path=None, name=None
    ) -> "SegmentationDirInfo":
        image_dir = proto.coerce_to(
            proto.ImageInfoI, image_dir, lambda x: ImageDirInfo(Path(x))
        )
        atlas_dir = proto.coerce_to(
            proto.AtlasInfoI, atlas_dir, lambda x: AtlasDirInfo(Path(x))
        )
        if path is None:
            name = name or image_dir.path.name + " at " + atlas_dir.path.name
            prefix = fs.common_prefix(image_dir.path, atlas_dir.path) or Path()
            path = prefix / "segm" / name
        return cls(path, image_dir, atlas_dir)

    def hook_file(self, readonly=True) -> fs.File[_img_atlas_t]:
        return fs.File(
            self.config().path,
            self._read_hook,
            not_implemented if readonly else self._write_hook,
        )

    def _read_hook(self, path) -> _img_atlas_t:
        config = io.read_yaml(path)["general"]
        paths = config["image_folder"], config["ontology_folder"]
        paths = [self.path / Path(p).expanduser().resolve() for p in paths]
        return ImageDirInfo(paths[0]), AtlasDirInfo(paths[1])  # type: ignore

    def _write_hook(self, path, value: _img_atlas_t) -> None:
        image_info, atlas_info = [
            os.fspath(fs.super_relative(self.path, Path(p)) or p) for p in value
        ]
        config = io.read_text(path)
        # this is kind of stupid that i made absolute paths the default
        config = config.replace(
            "replace with image folder absolute path", image_info, 1
        )
        config = config.replace(
            "replace with ontology folder absolute path", atlas_info, 1
        )
        io.write_text(path, config)


# </editor-fold>


# <editor-fold description="pipeline implementations">


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
        (imd.config().path, imd.tag_file().path), *(d.iterdir() for d in dirs)
    )
    src = imd.path
    zip_file = zip_file or (str(src) + ".backup.zip")
    io.zip_files(zip_file, files, src)


# </editor-fold>


# <editor-fold description="collect metadata">

register_error("PRE_METADATA_NOT_FOUND")
register_error("PRE_METADATA_READ_ERROR")


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


class collect_image_metadata(NamedTuple):  # this is a good example of how final pipelines should look like
    pre_meta_reader: _pre_meta_reader_t

    @staticmethod
    def from_ImageDirInfo(
        _, config: Union[ImageDirInfo, fs.File[dict], dict]
    ) -> "collect_image_metadata":
        if isa(config, ImageDirInfo):
            d = config.config().read()
        elif isa(config, fs.File):
            d = config.read()
        else:
            d = config
        r = _pre_meta_reader(d["metadata_keys"], d["file_name_fields"])
        return collect_image_metadata(r)

    @classmethod
    def get_main_loop(
        cls, image_dir: os.PathLike
    ) -> Tuple[Fn[[str], Opt[Err]], Iterable[str]]:
        image_dir = ImageDirInfo.read_from(image_dir)
        # image_dir_config = image_dir.config().read()
        self = cls.from_ImageDirInfo(None, image_dir)

        metas = image_dir.metadata()
        keys = image_dir.keys()
        pre_meta_dir = image_dir.pre_metadata_dir()
        read_pre_meta = self.pre_meta_reader

        def main_loop(k: str) -> Opt[Err]:
            d = read_pre_meta(pre_meta_dir, k)
            return d if is_err(d) else metas.__setitem__(k, d)

        return main_loop, keys

    @classmethod
    def main(cls, image_dir: os.PathLike, soft_err: _soft_err_t = None):
        loop, keys = cls.get_main_loop(image_dir)
        soft_err = DefaultHandlers.soft_error if soft_err is None else soft_err
        for k in keys:
            e = loop(k)
            if is_err(e):
                soft_err(e)


config_utils.construct_from_disp.register(
    (ImageDirInfo, collect_image_metadata),
)(collect_image_metadata.from_ImageDirInfo)

# </editor-fold>


# <editor-fold descr="raw images to npy">
register_error("RAW_IMAGES_TO_NUMPY")


@explain_error.register(_E.RAW_IMAGES_TO_NUMPY)
def explain_error(_, k, exc_type, exc_args) -> str:
    return f"pipeline: {raw_images_to_npy.__name__}, step: {k}, cause: exception {exc_type}{exc_args}"


def raw_images_to_npy(
    image_dir: os.PathLike, dtype="<i4", shape=(256, 256), soft_err: _soft_err_t = None
):
    image_dir = ImageDirInfo.read_from(image_dir)
    raw = image_dir.raw_images(dtype, shape)
    imgs = image_dir.images()
    soft_err = DefaultHandlers.soft_error if soft_err is None else soft_err
    for k in raw.keys():
        try:
            imgs[k] = raw[k]
        except Exception as e:
            soft_err(Err((_E.RAW_IMAGES_TO_NUMPY, k, type(e), e.args)))



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

        atlas_dir = proto.coerce_to(
            proto.AtlasInfoI, atlas_dir, AtlasDirInfo.read_from
        )
        save_path = atlas_dir.path / "slice_ids.txt"
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
        atlas_dir = proto.coerce_to(
            proto.AtlasInfoI, atlas_dir, AtlasDirInfo.read_from
        )
        ...

    @staticmethod
    def svgs(atlas_dir: os.PathLike):
        atlas_dir = proto.coerce_to(
            proto.AtlasInfoI, atlas_dir, AtlasDirInfo.read_from
        )
        ...


# </editor-fold>


# <editor-fold description="prerender masks">


def prerender_masks(atlas_dir: os.PathLike):
    atlas_dir = AtlasDirInfo.read_from(atlas_dir)


# </editor-fold>


# <editor-fold description="plot the masks">
def __get_frame_from_image_name(
    image_name: str,
) -> str:  # TODO: what to do with such things?
    return image_name.rsplit("_", maxsplit=1)[1]


class MaskPlotsDir:
    """represents directory with mask plots. if ``use_subroot``, do not use root directly"""

    def __init__(self, root, use_subroot=False):
        root = Path(root)
        self.root_dir = root / "masks_plots" if use_subroot else root
        self.formatter = fs.PrefixSuffixFormatter(self.root_dir, ".png")
        self.__formatter = "format plot names with this"

    __repr__ = fs.repr_DynamicDirInfo


@overload
def plot_the_masks(
    segmentation_dir,
    *,
    image_names: List[str] = None,
    structure_names: List[str] = None,
):
    ...


@overload
def plot_the_masks(
    image_dir,
    atlas_dir,
    out_path,
    *,
    image_names: List[str] = None,
    structure_names: List[str] = None,
):
    ...


def plot_the_masks(
    image_dir,
    atlas_dir=None,
    out_path=None,
    *,
    image_names: List[str] = None,
    structure_names: List[str] = None,
):
    if atlas_dir is None:  # call signature 1
        segm_dir = proto.coerce_to(
            proto.SegmentationInfoI, image_dir, SegmentationDirInfo.read_from
        )
        image_dir = segm_dir.image_info
        atlas_dir = segm_dir.atlas_info
        plots_dir = MaskPlotsDir(segm_dir, use_subroot=True)
    else:  # call signature 2
        image_dir = proto.coerce_to(
            proto.ImageInfoI, image_dir, ImageDirInfo.read_from
        )
        atlas_dir = proto.coerce_to(
            proto.AtlasInfoI, atlas_dir, AtlasDirInfo.read_from
        )
        plots_dir = MaskPlotsDir(out_path)
    flush_plot = DefaultHandlers.flush_plot
    plots_dir.root_dir.mkdir(parents=True, exist_ok=True)
    masks_dir_table = atlas_dir.masks()
    ont_table = atlas_dir.ontologies()
    cropped_images = image_dir.cropped_images()
    frames = dict()  # caches
    onts = dict()
    for im_name, s_name in iterators.product(image_names, structure_names):
        # TODO: remake into namedtuple or like, factor out main_loop closure, make error handling (if needed)
        frame = __get_frame_from_image_name(
            im_name
        )  # this should probably become main_loop closure parameter
        ont: Ontology = onts.get(frame)
        if ont is None:
            ont: Ontology = onts.setdefault(frame, ont_table[frame])
        structure = ont.find(
            lambda s: s.acronym == s_name or s.name == s_name
        )  # predicate should be configurable

        masks_dir = frames.setdefault(frame, masks_dir_table[frame])
        mask = masks_dir[structure]
        image = cropped_images[im_name]
        save_path = plots_dir.formatter.format(f"{im_name} with {s_name}")

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image, cmap="gray")
        mask = mask | np.flip(mask, axis=1)
        ax.imshow(np.where(mask, mask, np.nan), cmap="Set1", alpha=0.2)
        flush_plot(fig, save_path)


# </editor-fold>


# <editor-fold description="data to series">


class SerialImagesDir:
    def __init__(self, img_dir):
        img_dir = proto.coerce_to(
            proto.ImageInfoI, img_dir, ImageDirInfo.read_from
        )
        self.path = img_dir.path / "serial_images"
        self.index_file = fs.File(
            self.path / "index.csv",
            self._shape_file_read,
            self._shape_file_write,
        )
        self.table = io.NpyDir(self.path)

    __repr__ = fs.repr_DynamicDirInfo

    @staticmethod
    def _shape_file_write(path, value):
        pd.DataFrame.to_csv(
            value, path, sep="\t", index=not isa(value.index, pd.RangeIndex)
        )

    @staticmethod
    def _shape_file_read(path):
        t = pd.read_csv(path, sep="\t")
        cols = list(t.columns)
        cols.remove("im_width")
        cols.remove("im_height")
        return t.set_index(cols)


class NameTuple(NamedTuple):
    hour: int
    animal: int
    frame: int

    @classmethod
    def parse(cls, s: str):
        _, hour, animal, frame = s.split("_")
        return cls(int(hour), int(animal), int(frame[1:]))


def image_dir_to_series(image_dir):
    image_dir = proto.coerce_to(
        proto.ImageInfoI, image_dir, ImageDirInfo.read_from
    )
    image_dir_ser = SerialImagesDir(image_dir)
    image_dir_ser.path.mkdir(exist_ok=True)
    ci_in = image_dir.cropped_images()
    ci_out = image_dir_ser.table
    keys = [(NameTuple.parse(k), k) for k in image_dir.cropped_images().keys()]
    keygroups = iterators.collect_by(keys, lambda n: (n[0].animal, n[0].frame))
    keygroups = {
        k: sorted(v, key=lambda n: n[0].hour) for k, v in keygroups.items()
    }
    shapes = []
    # TODO: split loop, make lambdas into configuration
    get_shape = lambda g, sh: dict(
        animal=g[0], frame=g[1], im_width=sh[0], im_height=sh[1]
    )
    for group, keys in keygroups.items():
        images = np.stack([ci_in[name] for _, name in keys], axis=0)
        shapes.append(get_shape(group, images.shape[1:]))
        images = images.reshape((len(images), -1)).T
        ci_out["_".join(str(g) for g in group)] = images
    shapes = pd.DataFrame(shapes)
    image_dir_ser.index_file.write(shapes)


@overload
def plot_the_masks_image_dir_serial(
    segmentation_dir,
    *,
    image_names: List[str] = None,
    structure_names: List[str] = None,
):
    ...


@overload
def plot_the_masks_image_dir_serial(
    image_dir,
    atlas_dir,
    out_path,
    *,
    image_names: List[str] = None,
    structure_names: List[str] = None,
):
    ...


def plot_the_masks_image_dir_serial(
    image_dir,
    atlas_dir=None,
    out_path=None,
    *,
    image_names: List[str] = None,
    structure_names: List[str] = None,
):
    if atlas_dir is None:  # call signature 1
        segm_dir = proto.coerce_to(
            proto.SegmentationInfoI, image_dir, SegmentationDirInfo.read_from
        )
        image_dir = segm_dir.image_info
        atlas_dir = segm_dir.atlas_info
        plots_dir = segm_dir.path / "flat_mask_plots"
    else:  # call signature 2
        image_dir = proto.coerce_to(
            proto.ImageInfoI, image_dir, ImageDirInfo.read_from
        )
        atlas_dir = proto.coerce_to(
            proto.AtlasInfoI, atlas_dir, AtlasDirInfo.read_from
        )
        plots_dir = out_path / "flat_mask_plots"
    flush_plot = DefaultHandlers.flush_plot
    ser_im_dir = SerialImagesDir(image_dir)
    images = ser_im_dir.table
    onts = atlas_dir.ontologies()
    n_onts = len(list(onts.keys()))
    masks = atlas_dir.masks()
    get_ont = lru_cache(n_onts)(onts.__getitem__)
    get_masks_dir = lru_cache(n_onts)(masks.__getitem__)
    for structure_name, image_name in iterators.product(structure_names, image_names):
        ont_name = "f" + image_name.split("_")[1]  # to config
        ont = get_ont(ont_name)
        structure = ont.find(lambda s: s.name == structure_name or s.acronym == structure_name)
        mask = get_masks_dir(ont_name)[structure]
        img = images[image_name]
        mask_f = mask.flatten()
        masked = img[mask_f]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # to config
        ax.plot(masked, c="blue", alpha=0.2)  # to config
        ax.set_title(f"{structure_name} at {image_name}")
        flush_plot(fig, plots_dir / f"{image_name} {structure_name}")








# </editor-fold>

# <editor-fold description="">
# </editor-fold>

# <editor-fold description="average cropped images">
# </editor-fold>

# </editor-fold>
