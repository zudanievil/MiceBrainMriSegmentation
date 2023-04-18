"""
common imports and definitions
used as `from prelude import *`
"""
import sys
import os
from typing import (
    TypeVar as _TV,
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
)
from pathlib import Path


T = _TV("T")
T1 = _TV("T1")
T2 = _TV("T2")
K = _TV("K")
V = _TV("V")


class Version(NamedTuple):
    major: int
    minor: int
    patch: int


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
    """:return type(x) == Err"""
    return type(x) == Err


def do_it(f: Fn[[], T]) -> T:
    """return f() # for complex object initialization"""
    return f()


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


@do_it
def ipyformat() -> "IPython.core.formatters.PlainTextFormatter":
    from IPython.core.formatters import PlainTextFormatter as _PTF

    return _PTF()


def not_implemented(*_, **__):
    raise NotImplementedError
