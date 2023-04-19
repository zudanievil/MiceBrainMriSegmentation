"""
dataset interfaces
"""
from ..prelude import *
PathLike = os.PathLike

__all__ = [
    "File",
    "FileTable",
    "FileFactory",
    "is_empty_dir",
]

_read_t = Fn[[PathLike], T]
_write_t = Fn[[PathLike, T], None]
_format_t = Fn[[K], PathLike]
_unformat_t = Fn[[PathLike], K]
_keys_t = Fn[[], Iterable[K]]


# def _meth_fmt(obj, *methods: str) -> List[str]:
#     """for formatting callable fields"""
#     ret = []
#     for meth in methods:
#         fn = getattr(obj, meth)
#         if fn == not_implemented:
#             continue
#         ret.append(f"{meth} = {ipyformat(fn)}")
#     return ret


class File(Generic[T]):
    __slots__ = "path", "_read", "_write",

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
            return name_suf[:-len(self.suffix)] if self.suffix else name_suf

    def keys(self) -> Iterator[str]:
        no_suffix = not self.suffix
        for p in self.prefix.iterdir():
            if not p.is_file():
                continue
            if no_suffix:
                yield p.name
                continue
            ps = str(p)
            if ps.endswith(self.suffix):
                yield ps[:-len(self.suffix)]

    def to_FileTable(self, read: _read_t = not_implemented, write: _write_t = not_implemented) -> FileTable[str, T]:
        return FileTable(self.format, self.unformat, self.keys, read, write)


def is_empty_dir(path: Path) -> bool:
    for _ in path.iterdir():
        return False
    return True

