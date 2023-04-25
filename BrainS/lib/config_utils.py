"""
Some useful configuration tools.
* `get_from_dict_constructor` -- object construction from dict

"""
from ..prelude import *
from .functional import (
    ValDispatch,
    Classifier,
)


__all__ = [
    "from_dict_disp",
    "is_named_tuple",
    "NT_from_dict",
    "NT_replace",
    "get_from_dict_constructor",
    "_from_dict_interface_",
]


def is_named_tuple(t: type) -> bool:
    """True if class was created by `typing.NamedTuple` or similar method"""
    return hasattr(t, "_fields") and t.__bases__ == (tuple,)


NT = TypeVar("NT", bound=NamedTuple)


def NT_replace(x: NT, **fields) -> NT:
    """return a shallow copy with specified fields overwritten"""
    d = x._asdict()
    d.update(fields)
    return x.__class__(**d)


def NT_from_dict(t: Type[NT], d: dict) -> NT:
    """non-strict version of `t(**d)`. does not error when extra keys are present in dict"""
    d1 = dict()
    for f in t._fields:
        v = d.get(f)
        if v is not None:
            d1[f] = v
    return t(**d1)


_from_dict_t = Fn[[Type[T], dict], T]


def _from_dict_interface_(t: Type[T], d: dict) -> T:
    """uses `t._from_dict_interface_(d)` classmethod"""
    return t._from_dict_interface_(d)


@Classifier.new()
def get_from_dict_constructor(t: Type[T]) -> _from_dict_t:
    """get an appropriate ``form_dict(t: Type[T], d: dict) -> T`` function for a given type"""
    if hasattr(t, "_from_dict_interface_"):
        return _from_dict_interface_


@get_from_dict_constructor.add_arm()
def __named_tuple(t: type) -> _from_dict_t:
    if is_named_tuple(t):
        return NT_from_dict


@ValDispatch.new(object)
def from_dict_disp(t: Type[T], d: dict) -> T:
    """functional interface for constructing objects from dictionary"""
    raise NotImplementedError


@get_from_dict_constructor.add_else
def __multi_dispatch(*_) -> _from_dict_t:
    return from_dict_disp


# TODO: add `to dict` interface
