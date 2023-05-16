"""
common imports and definitions
used as `from prelude import *`
"""
import sys
import os
from typing import (
    TypeVar,
    Callable as Fn,
    Optional as Opt,
    Protocol as Proto,
    List,
    Dict,
    Tuple,
    Union,
    Generic,
    NamedTuple,
    Type,
    Any,
    Iterable,
    Iterator,
)
from pathlib import Path
from . import __config as cfg


T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
K = TypeVar("K")
V = TypeVar("V")

# <editor-fold descr="neat unpythonic stuff">


class Err(Exception):
    """This exception will be used in a very unpythonic way"""

    __slots__ = ("data",)

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"Err(data={repr(self.data)})"

    __str__ = __repr__


def is_err(x) -> bool:
    """:return x.__class__ is Err"""
    return x.__class__ is Err


def isa(x, cls) -> bool:
    """:return x.__class__ is cls"""
    return x.__class__ is cls


def raise_(error) -> None:
    raise error


def identity(x: T) -> T:
    return x


def setitem(obj, key, value):
    obj[key] = value


def foreach(itr: Iterable) -> None:
    for _ in itr:
        pass


def none(*_, **__) -> None:
    return None


def not_implemented(*_, **__):
    raise NotImplementedError


def do_it(f: Fn[[], T]) -> T:
    """return f() # for complex object initialization"""
    return f()


# </editor-fold>


# <editor-fold descr="all things meta">


def exec_(code, frame: int = 0):
    """
    frame = 0 -- execute code in this frame
    frame >= 1 -- further up the stack
    """
    frame += 1
    exec(code, sys._getframe(frame).f_globals, sys._getframe(frame).f_locals)


def include(p: Union[Path, str], frame: int = 0):
    """
    include file by "code copy-pasting" (as #include in C)
    frame = 0 to include at call site, frame >= 1 to include further up the stack
    """
    with open(p, "rt") as f:
        src = f.read()
    code = compile(src, str(p), mode="exec")
    exec_(code, frame + 1)


def import_(path, *, register=False) -> "module":
    """import module by path. optionally register in `sys.modules`"""
    from importlib import util

    name = Path(path).stem
    spec = util.spec_from_file_location(name, path)
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if register:
        sys.modules[name] = mod
    return mod


# </editor-fold>


# <editor-fold descr="formatting">


@do_it
def ipyformat() -> "IPython.core.formatters.PlainTextFormatter":
    from IPython.core.formatters import PlainTextFormatter as _PTF

    return _PTF()


def repr_slots(obj) -> str:  # pretty slow !40-50 µs, but generic
    """pretty-print an object with __slots__"""
    assert hasattr(obj, "__slots__"), f"{obj} does not have '__slots__'"
    cls = obj.__class__
    cls_name = f"{cls.__module__}.{cls.__name__}"
    slots = obj.__slots__

    fslots = []
    for name in slots:
        value = getattr(obj, name)
        if (value is not_implemented) or (value is ...):
            continue
        fslots.append(f"{name} = {ipyformat(value)}")
    fslots = ",\n\t".join(fslots)
    return f"{cls_name}(\n\t{fslots}\n)"


# </editor-fold>


# <editor-fold descr="more unpythonic stuff">


class Box(Generic[T]):
    __slots__ = ("data",)

    def __init__(self, data: T = None):
        self.data = data

    __repr__ = repr_slots


# </editor-fold>