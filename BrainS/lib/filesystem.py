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
    return p0.parents[i]


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
