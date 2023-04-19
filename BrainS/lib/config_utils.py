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
    "named_tuple_from_dict",
    "get_from_dict_constructor",
]


@ValDispatch.new(object)
def from_dict_disp(t: type, d: dict) -> object:
    """functional interface for constructing object from dictionary"""
    raise NotImplementedError


def is_named_tuple(t: type) -> bool:
    """True if class was created by `typing.NamedTuple` or similar method"""
    return hasattr(t, "_fields") and t.__bases__ == (tuple,)


def named_tuple_from_dict(t: Type[T], d: dict) -> T:
    """non-strict version of `t(**d)`. does not error when extra keys are present in dict"""
    d1 = dict()
    for f in t._fields:
        v = d.get(f)
        if v is not None:
            d1[f] = v
    return t(**d1)


_from_dict_t = Fn[[Type[T], dict], T]


def __from_dict_interface(t: Type[T], d: dict) -> T:
    """uses `t._from_dict_interface_(d)` classmethod"""
    return t._from_dict_interface_(d)


@Classifier.new()
def get_from_dict_constructor(t: Type[T]) -> _from_dict_t:
    """get an appropriate `form_dict` constructor for a type"""
    if hasattr(t, "_from_dict_interface_"):
        return __from_dict_interface


@get_from_dict_constructor.add_arm()
def __named_tuple(t: type) -> _from_dict_t:
    if is_named_tuple(t):
        return named_tuple_from_dict


@get_from_dict_constructor.add_else
def __multi_dispatch(*_) -> _from_dict_t:
    return from_dict_disp
