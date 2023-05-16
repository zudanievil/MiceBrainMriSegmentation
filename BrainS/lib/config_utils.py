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
    """apply ``construct_from_disp``"""
    return construct_from_disp((type(x), t1), x)


def get_constructor(src_t: Type[T], dst_t: Type[T1]) -> Opt[Fn[[T], T1]]:
    """search ``construct_from_disp`` registry"""
    return construct_from_disp.registry.get((src_t, dst_t))


class DocumentedNamespace:
    """
    similar to argparse.Namespace, but upgraded for interactive use.
    stores type hints and documentation and pretty-prints them with __repr__()
    ``
    important = DocumentedNamespace(_name="important")
    important.list = [1, 2, 3]
    important._list_type = List[int]
    important._list_doc = "be sure to keep this list available"
    print(important) # try it!
    ``
    """

    def __init__(self, _name: str = None, _doc: str = None, **kwargs):
        self._name = _name
        self._doc = _doc
        self.__dict__.update(kwargs)

    def __setattr__(self, k: str, v):
        if k.startswith("_") and k not in ("_name", "_doc"):
            k2 = k.removeprefix("_").removesuffix("_doc").removesuffix("_type")
            if k2 not in self.__dict__:
                raise KeyError(f"no attribute named {k2} found in __dict__")
        self.__dict__[k] = v

    def _get_doc_(self, attr_name):
        return self.__dict__.get(f"_{attr_name}_doc")

    def _get_type_(self, attr_name):
        return self.__dict__.get(f"_{attr_name}_type")

    def __repr__(self):
        parts = [f"{self.__class__.__name__}"]
        if self._name:
            parts.append(f"[name]: '{self._name}'")
        if self._doc:
            parts.append(f"[doc]: {self._doc}")
        parts.append("attributes:")
        parts.append("=" * 10)
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            parts.append(f".{k}:")
            doc = self._get_doc_(k)
            typ = self._get_type_(k)
            if typ is not None and isa(typ, type):
                typ = typ.__name__
                parts.append(f"[type]: {typ}")
            parts.append(f"[value]: {ipyformat(v)}")
            if doc is not None:
                parts.append(f"[doc]: {doc}")
            parts.append("-" * 5)
        parts.append("=" * 10)
        return "\n".join(parts)
