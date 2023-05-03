"""
Some useful configuration tools.
* `get_from_dict_constructor` -- object construction from dict

"""
from ..prelude import *
from .functional import (
    ValDispatch,
)


__all__ = [
    "is_named_tuple",
    "NT_from_dict",
    "NT_replace",
    "get_constructor",
    "construct_from",
    "construct_from_disp",
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


@ValDispatch.new((None, None))
def construct_from_disp(from_to: Tuple[Type[T], Type[T1]], config: T) -> T1:
    """derive value of one type from the value of another"""
    return None


def construct_from(t1: Type[T1], x: T) -> T1:
    """apply ``construct_from_disp`` """
    return construct_from_disp((type(x), t1), x)


def get_constructor(src_t: Type[T], dst_t: Type[T1]) -> Opt[Fn[[T], T1]]:
    """search ``construct_from_disp`` registry"""
    return construct_from_disp.registry.get((src_t, dst_t))

