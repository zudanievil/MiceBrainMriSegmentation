"""
dataset interfaces
"""
from ..prelude import *


__all__ = [
    "File",
    "FileTable",
]

_read_t = Fn[[Path], T]
_write_t = Fn[[Path, T], None]
_format_t = Fn[[K], Path]
_unformat_t = Fn[[Path], K]


def _meth_fmt(obj, *methods: str) -> List[str]:
    """for formatting callable fields"""
    ret = []
    for meth in methods:
        fn = getattr(obj, meth)
        if fn == not_implemented:
            continue
        ret.append(f"{meth} = {ipyformat(fn)}")
    return ret


class File(Generic[T]):
    __slots__ = "_read", "_write", "path"

    def __init__(
        self,
        path: Path,
        _read: _read_t = not_implemented,
        _write: _write_t = not_implemented,
    ):
        self.path = path
        self._read = _read
        self._write = _write

    def __fspath__(self):
        return str(self.path)

    def read(self) -> T:
        return self._read(self.path)

    def write(self, x: T):
        return self._write(self.path, x)

    def add_read(self, _read: _read_t) -> None:
        self._read = _read

    def add_write(self, _write: _write_t) -> None:
        self._write = _write

    def __repr__(self):
        m = ",\n\t".join(_meth_fmt(self, "_read", "_write"))
        return f"{self.__class__.__name__}({self.path},\n\t{m}\n)"


class FileTable(Generic[K, T]):
    """
    Typically, files in datasets can be uniquely identified by some combination of
    string literals and numbers, like `("axial", 12, "placebo")` if
    """

    __slots__ = "format", "unformat", "_read", "_write"

    def __init__(
        self,
        format: _format_t = not_implemented,
        unformat: _unformat_t = not_implemented,
        _read: _read_t = not_implemented,
        _write: _write_t = not_implemented,
    ):
        self.format = format
        self.unformat = unformat
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

    def file(self, key: K) -> File[T]:
        p = self.format(key)
        return File(p, self._read, self._write)

    def __getitem__(self, key: K) -> T:
        return self._read(self.format(key))

    def __setitem__(self, key: K, value: T) -> None:
        return self._write(self.format(key), value)

    def get_key(self, f: Union[File, Path]) -> tuple:
        p = f.path if type(f) == File else f
        return self.unformat(p)

    def __repr__(self):
        m = ",\n\t".join(
            _meth_fmt(self, "format", "unformat", "_read", "_write")
        )
        return f"{self.__class__.__name__}(\n\t{m}\n)"


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

    def __repr__(self):
        m = ",\n\t".join(_meth_fmt(self, "read", "write"))
        return f"{self.__class__.__name__}(\n\t{m}\n)"

    def __call__(self, file: Path) -> File[T]:
        return File(file, self.read, self.write)
