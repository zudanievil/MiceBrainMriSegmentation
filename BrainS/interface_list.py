"""
lists of interfaces for the reference
(plain generic functions not included)

Lists
----------

TYPE_CLASSES
Type classes are highly generic FP-styled interfaces,
that simply batch together functions of certain types.
Additionally, they can promise some sort of properties that hold
for each instance or provide useful ways to compose those.

DISPATCHED
Function-like objects that can be dynamically changed
(use some sort of dispatch)
or functions that rely on class metadata like ``__slots__``.

PROTOCOLS
Protocol classes and functions that
use python structural typing (duck typing)
"""

from .lib.functional import Classifier, ValDispatch, Dispatch
from .lib.filesystem import FileTable, File, FileFactory
from .lib.iterators import (
    Flattener,
    Unflattener,
    flat_tree_lift,
    flat_tree_filter_lift,
    repr_flat,
)
from .lib.io_utils import file
from .lib.config_utils import (
    from_dict_disp,
    get_from_dict_constructor,
    NT_replace,
    NT_from_dict,
    _from_dict_interface_,
)
from .prelude import repr_slots
from .protocols import InfoI, ImageInfoI, AtlasInfoI, SegmentationInfoI

__all__ = [
    "TYPE_CLASSES",
    "DISPATCHED",
    "PROTOCOLS",
    "ALL",
]

TYPE_CLASSES = [
    Classifier,
    ValDispatch,
    Dispatch,
    File,
    FileTable,
    FileFactory,
    Flattener,
    Unflattener,
    flat_tree_lift,
    flat_tree_filter_lift,
]

DISPATCHED = [
    file,
    from_dict_disp,
    get_from_dict_constructor,
    NT_replace,
    NT_from_dict,
    repr_slots,
    repr_flat,
]

PROTOCOLS = [
    _from_dict_interface_,
    InfoI,
    ImageInfoI,
    AtlasInfoI,
    SegmentationInfoI,
]

ALL = TYPE_CLASSES + PROTOCOLS + DISPATCHED
