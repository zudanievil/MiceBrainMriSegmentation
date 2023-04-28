from .prelude import *
from .lib import filesystem as fs, io_utils as io
from ._version import __version__

__all__ = [
    "InfoI",
    "ImageInfoI",
    "AtlasInfoI",
    "SegmentationInfoI",
    "implements",
    "get_implementations",
]


class InfoDirTag(NamedTuple):
    class_tag: str
    kwargs: dict

    @classmethod
    def new(cls, class_tag: str, **kwargs) -> "InfoDirTag":
        return cls(class_tag, kwargs)


TagFileFactory = fs.FileFactory()


@TagFileFactory.add_read
def __(path) -> InfoDirTag:
    return InfoDirTag.new(**io.read_toml(path))


@TagFileFactory.add_write
def __(path, value: InfoDirTag) -> None:
    d = value.kwargs.copy()
    d[value._fields[0]] = value.class_tag
    io.write_toml(path, d)


class InfoI(Proto):
    TAG: str = ""
    """unique string identifier of a class. should be persistent over versions"""
    TAG_FILE_NAME = ".BrainS_directory_info.toml"

    def tag(self) -> InfoDirTag:
        """create a tag object for writing"""
        return InfoDirTag.new(
            self.TAG, about=cfg.about, version=cfg.__version__
        )

    def tag_file(self) -> fs.File[InfoDirTag]:
        """create fs.File to mark directory"""
        return TagFileFactory(self.path() / self.TAG_FILE_NAME)

    @classmethod
    def read_from(cls, directory: os.PathLike, **options) -> "InfoI":
        """
        This should be a robust constructor, that pipelines utilise.
        The idea is that this constructor can produce some sort of validation,
        at the very minimum it should assert that the directory exists.
        Potentially, it can do io and stuff.
        Should short-circuit (return shallow copy, omit validation) if ``type(directory) == cls``
        """
        ...

    def path(self) -> Path:
        ...

    def __fspath__(self) -> str:
        ...

    def config(self) -> fs.File:
        ...

    def default_config(self) -> fs.File:
        ...

    def __repr__(self) -> str:
        ...

    def create(self) -> None:
        ...


class ImageInfoI(InfoI):
    def images(self) -> fs.FileTable:
        ...

    def metadata(self) -> fs.FileTable:
        ...

    def cropped_images(self) -> fs.FileTable:
        ...

    def image_dir(self) -> Path:
        ...

    def metadata_dir(self) -> Path:
        ...

    def cropped_image_dir(self) -> Path:
        ...


class AtlasInfoI(InfoI):
    def svgs(self) -> fs.FileTable:
        ...

    def ontologies(self) -> fs.FileTable:
        ...

    def masks(self) -> fs.FileTable:
        ...


class SegmentationInfoI(InfoI):
    def atlas_info(self) -> AtlasInfoI:
        ...

    def image_info(self) -> ImageInfoI:
        ...

    def temp_dir(self) -> Path:
        ...

    def plots_dir(self) -> Path:
        ...

    def tables_dir(self) -> Path:
        ...


# =========== some shared functionality ======


# =========== for introspection =====


__protocol_special_keys = {
    "__dict__",
    "__weakref__",
    "__parameters__",
    "_is_protocol",
    "__subclasshook__",
    "__init__",
    "__annotations__",
    "__abstractmethods__",
    "_abc_impl",
}

__protocol_implementations = dict()


def get_implementations(p: "Type[Proto]") -> set:
    """show protocol implementations set (new classes are added with `@implements`)"""
    return __protocol_implementations[p]


def implements(
    p: "Type[Proto]", t: Type[T] = None
) -> Opt[Fn[[Type[T]], Type[T]]]:
    """
    register implementation. can be used as function or as a decorator:
    ```
    @implements(MyProtocol)
    class MyImplementation: ...
    ```
    """

    def clj(t: Type[T]) -> Type[T]:
        impls = __protocol_implementations.setdefault(p, set())
        if t in impls:
            return t

        missing = []
        for k in p.__dict__.keys():
            if k not in t.__dict__ and k not in __protocol_special_keys:
                missing.append(k)
        if missing:
            raise KeyError(f"missing methods: {missing}")

        impls.add(t)
        return t

    if t is None:  # as decorator
        return clj
    clj(t)  # as function call


def coerce_to(protocol: Type[T], x, alt_cons) -> T:
    """
    if ``type(x)`` is in the ``protocol`` implementation list,
    ``return x``. else apply ``alt_cons`` to x
    """
    is_impl = type(x) in get_implementations(protocol)
    return x if is_impl else alt_cons(x)


# ================
