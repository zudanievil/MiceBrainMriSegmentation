"""
dataset interfaces
"""

from ..prelude import *
from .iterators import Flattener

PathLike = os.PathLike

__all__ = [
    "File",
    "FileTable",
    "FileFactory",
    "PrefixSuffixFormatter",
    "is_empty_dir",
    "common_prefix",
    "common_parts",
    "super_relative",
    "iter_tree",
    "iter_tree_braces",
    "repr_DynamicDirInfo",
]

_read_t = Fn[[PathLike], T]
_write_t = Fn[[PathLike, T], None]
_format_t = Fn[[K], PathLike]
_unformat_t = Fn[[PathLike], K]
_keys_t = Fn[[], Iterable[K]]


class File(Generic[T]):
    __slots__ = (
        "path",
        "_read",
        "_write",
    )

    def __init__(
        self,
        path: PathLike,
        _read: _read_t = not_implemented,
        _write: _write_t = not_implemented,
    ):
        self.path = path
        self._read = _read
        self._write = _write

    def __fspath__(self):
        return os.fspath(self.path)

    def read(self) -> T:
        return self._read(self.path)

    def write(self, x: T):
        return self._write(self.path, x)

    def add_read(self, _read: _read_t) -> None:
        self._read = _read

    def add_write(self, _write: _write_t) -> None:
        self._write = _write

    __repr__ = repr_slots


class FileTable(Generic[K, T]):
    """
    Typically, files in datasets can be uniquely identified by some combination of
    string literals and numbers, like `("axial", 12, "placebo")` if
    """

    __slots__ = "format", "unformat", "keys", "_read", "_write"

    def __init__(
        self,
        format: _format_t = not_implemented,
        unformat: _unformat_t = not_implemented,
        keys: _keys_t = not_implemented,
        _read: _read_t = not_implemented,
        _write: _write_t = not_implemented,
    ):
        self.format = format
        self.unformat = unformat
        self.keys = keys
        self._read = _read
        self._write = _write

    def add_format(self, format: _format_t) -> None:
        self.format = format

    def add_unformat(self, unformat: _unformat_t) -> None:
        self.unformat = unformat

    def add_read(self, _read: _read_t) -> None:
        self._read = _read

    def add_write(self, _write: _write_t) -> None:
        self._write = _write

    def add_keys(self, keys: _keys_t) -> None:
        self.keys = keys

    def file(self, key: K) -> File[T]:
        p = self.format(key)
        return File(p, self._read, self._write)

    def __getitem__(self, key: K) -> T:
        return self._read(self.format(key))

    def __setitem__(self, key: K, value: T) -> None:
        return self._write(self.format(key), value)

    def get_key(self, f: Union[File, PathLike]) -> K:
        p = f.path if type(f) == File else f
        return self.unformat(p)

    __repr__ = repr_slots


class FileFactory(Generic[T]):
    __slots__ = "read", "write"

    def __init__(
        self,
        read: _read_t = not_implemented,
        write: _write_t = not_implemented,
    ):
        self.read = read
        self.write = write

    def add_read(self, _read: _read_t) -> None:
        self.read = _read

    def add_write(self, _write: _write_t) -> None:
        self.write = _write

    __repr__ = repr_slots

    def __call__(self, file: PathLike) -> File[T]:
        return File(file, self.read, self.write)


class PrefixSuffixFormatter(NamedTuple):
    """
    commonly datasets are just folders with files of certain type:
    `prefix/(name + suffix)`
    suffix can be any valid part of a file name (does not have to start with ".")
    empty suffix means that no filtering on suffix is done and nothing is added to file name
    """

    prefix: Path
    suffix: str = ""

    def format(self, name) -> Path:
        return self.prefix / (name + self.suffix)

    def unformat(self, path: os.PathLike) -> str:
        if isinstance(path, Path):
            return path.stem
        else:
            name_suf = str(path).rsplit(os.pathsep, 1)[1]
            return name_suf[: -len(self.suffix)] if self.suffix else name_suf

    def keys(self) -> Iterator[str]:
        no_suffix = not self.suffix
        for p in self.prefix.iterdir():
            if not p.is_file():
                continue
            pn = p.name
            if no_suffix:
                yield pn
            elif pn.endswith(self.suffix):
                yield pn[: -len(self.suffix)]

    def to_FileTable(
        self, read: _read_t = not_implemented, write: _write_t = not_implemented
    ) -> FileTable[str, T]:
        return FileTable(self.format, self.unformat, self.keys, read, write)


def is_empty_dir(path: Path) -> bool:
    for _ in path.iterdir():
        return False
    return True


def common_parts(p0: Path, p1: Path) -> Opt[int]:
    """:return None if paths don'ptr have same anchor else i such that
    `p0.parts[:i]` (`== p1.parts[:i]`) is the common part of both paths"""
    if p0.anchor != p1.anchor:
        return None
    pts0 = p0.parts
    pts1 = p1.parts
    i = 0
    for i, (pt0, pt1) in enumerate(zip(pts0, pts1)):
        if pt0 != pt1:
            break
    return i


def super_relative(parent: Path, child: Path) -> Opt[Path]:
    """
    :return `p`, such that `(parent / p).resolve() == child`
    return None if paths have different anchors
    """
    i = common_parts(parent, child)
    if i is None:
        return None
    pts_c = child.parts
    up = len(parent.parts) - i
    return Path("../" * up + "/".join(pts_c[i:]))


def common_prefix(p0: Path, p1: Path) -> Opt[Path]:  # TODO: test
    i = common_parts(p0, p1)
    if i is None:
        return None
    parents = p0.parents
    return parents[len(parents) - i]


def __dir_has_children(x):
    return type(x) != Box and x.is_dir()


def __dir_head_tail_iter(d: Path) -> Iterator[Box[Path]]:
    yield Box(d)  # shield head
    yield from d.iterdir()


def __dir_unwrap(d: Union[Path, Box[Path]]) -> Path:
    return d.data if type(d) == Box else d


iter_tree = Flattener(
    braces=False,
    has_children=__dir_has_children,
    get_children=__dir_head_tail_iter,
    leaf_unwrap=__dir_unwrap,
)
iter_tree_braces = Flattener(
    braces=True,
    has_children=__dir_has_children,
    get_children=__dir_head_tail_iter,
    leaf_unwrap=__dir_unwrap,
)


class FilterFormatter(NamedTuple):
    """
    if a datataset has complex file hierarchy,
    it may make sense to simply filter files by predicate
    """

    prefix: Path
    filter: Fn[[Path], bool]

    def format(self, path: str) -> Path:
        return self.prefix / path

    def unformat(self, path: PathLike) -> str:
        path = path if isinstance(path, Path) else Path(path)
        return str(path.relative_to(self.prefix))  # type: ignore

    def keys(self) -> Iterator[str]:
        return (
            str(p.relative_to(self.prefix))
            for p in iter_tree(self.prefix)
            if self.filter(p)
        )

    def to_FileTable(
        self, read: _read_t = not_implemented, write: _write_t = not_implemented
    ) -> FileTable[str, T]:
        return FileTable(self.format, self.unformat, self.keys, read, write)


class TableFormatter(NamedTuple):
    """
    just have a lookup table for paths
    """

    table: Dict[K, Path]

    def format(self, key: K) -> Path:
        return self.table[key]

    def unformat(self, path):
        path = Path(path)
        for k, v in self.table.items():
            if v == path:
                return k
        else:
            raise ValueError(path)

    def keys(self) -> Iterable[K]:
        return self.table.keys()

    __repr__ = object.__repr__  # because it will be a freaking huge output

    def to_FileTable(
        self, read: _read_t = not_implemented, write: _write_t = not_implemented
    ) -> FileTable[str, T]:
        return FileTable(self.format, self.unformat, self.keys, read, write)


def repr_DynamicDirInfo(self):
    """
    represent an object that conforms to a special standard:
    ```
    class ExampleProjectDir:
        "info for my project directory"
        def __init__(self, root: Path):
            self.plots_dir = root / "plots"
            # .__x will be a docstring for .x
            self.__plots_dir = "a directory for plots"
            self.cfg = TomlFile(root / "run-cfg.toml")
            self.__cfg = "configuration for a pipeline"

        __repr__ = repr_DynamicDirInfo
    ```
    Displays additional info for PathLike fields.
    Class must have ``__init__`` method and
    instances must use ``__dict__``
    """
    cls = self.__class__
    cls_name = cls.__name__
    parts = [
        f"\n[About {cls.__name__}]",
    ]
    if cls.__doc__:
        parts.append(f"[Doc]: {cls.__doc__}")
    parts.append(f"[Init]: {ipyformat(cls.__init__)}")
    parts.append("[Attributes]:")

    parts = [
        "\n".join(parts),
    ]
    obj_dict = self.__dict__
    for k, v in obj_dict.items():
        if k.startswith("_"):
            continue
        try:
            p = Path(v)
            if not p.exists():
                status = "(not exist)"
            elif p.is_dir():
                status = "(dir)"
            else:
                status = ""
            path = f"[path]: {p} {status}"
        except TypeError:  # not an os.PathLike
            path = None
        attrib = f".{k}:"
        repr_ = None if isinstance(v, Path) else f"[repr]: {repr(v)}"
        doc = obj_dict.get(f"_{cls_name}__{k}")
        doc = None if doc is None else f"[doc]: {doc}"
        parts.append("\n".join(s for s in (attrib, path, doc, repr_) if s))
    parts.append(f"end of {cls.__name__};\n")
    return "\n_____________\n".join(parts)


"""
This ^^^ may seem as a useless function, however, I find myself making
classes like this quite often. It's nice to have some good support for 
them at runtime, while keeping their code declarative.
"""

#
# class FnameFieldsFormatter(NamedTuple):
#     prefix: Path
#     template: List[str, Tuple[str, Type[T]]]
#     cons: NamedTuple
#
#     @classmethod
#     def new(cls, prefix, template: List[str, Tuple[str, Type[T]]]):
#         nt = NamedTuple("generated", **{v[0]: v[1] for v in template if not isa(v, str)})
#         return cls(Path(prefix), template, nt)
#
#     def format(self, k: tuple):
#         self.prefix / ...
