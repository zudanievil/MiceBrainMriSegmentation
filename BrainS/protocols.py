from BrainS.prelude import *
from BrainS.lib import filesystem as fs

__all__ = [
    "InfoI",
    "ImageInfoI",
    "AtlasInfoI",
    "SegmentationInfoI",
    "impl",
    "impls",
]


def impl(p: Type[Proto], t: Type[T] = None) -> Opt[Fn[[Type[T]], Type[T]]]:
    """
    register implementation. can be used as function or as a decorator:
    ```
    @impl(MyProtocol)
    class MyImplementation: ...
    ```
    """

    def clj(t: Type[T]) -> Type[T]:
        missing = []
        for k in t.__dict__.keys():
            if k not in p.__dict__:
                missing.append(k)
        if missing:
            raise KeyError(f"missing methods: {missing}")

        if not hasattr(p, "__protocol_implementations__"):
            p.__protocol_implementations__ = []
        p.__protocol_implementations__.append(t)
        return t

    if t is None:  # as decorator
        return clj
    clj(t)  # as function call


def impls(p: Type[Proto]) -> List[type]:
    """show protocol implementation"""
    return p.__protocol_implementations__  # type: ignore


class InfoI(Proto):
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

    @classmethod
    def from_path(self, folder: Path) -> "InfoI":
        ...

    def create(self) -> "InfoI":
        ...


class ImageInfoI(InfoI):
    def images(self) -> fs.FileTable:
        ...

    def metadata(self) -> fs.FileTable:
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
