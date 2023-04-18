from BrainS.prelude import *
from BrainS.lib import filesystem as fs

__all__ = [
    "InfoI",
    "ImageInfoI",
    "AtlasInfoI",
    "SegmentationInfoI",
    "implements",
    "get_implementations",
]


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

    def cropped_images(self) -> fs.FileTable:
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


# =========== for introspection =====


__protocol_special_keys = {
    "__dict__",
    "__weakref__",
    "__parameters__",
    "_is_protocol",
    "__subclasshook__",
    "__init__",
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
