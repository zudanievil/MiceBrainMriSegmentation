"""
implements basic reader for analyze 7.5 format that sort of works.
for something more sophisticated you can use `nibabel` library: https://pypi.org/project/nibabel/
"""
import numpy as _np
from pathlib import Path as _Path
from typing import Any as _Any, Tuple as _Tup, Optional as _Opt
from numpy.typing import DTypeLike as _DTL, NDArray as _Arr

DATA_TYPE_CODES = {
    0: "unknown",
    1: "binary",
    2: _np.dtype("c"),  # unsigned char
    4: _np.dtype("<i2"),  # signed short
    8: _np.dtype("<i4"),  # signed int
    16: _np.dtype("<f4"),  # signed float
    32: _np.dtype("<c8"),  # complex
    64: _np.dtype("<f8"),
    128: "RGB",
    255: "all",
}
"""analyze7.5 ptr type literals, mapped onto more readable ones"""

HDR_NUMPY_DTYPE = _np.dtype(
    [
        ("sizeof_hdr", "<i4"),
        ("data_type", "S10"),
        ("db_name", "S18"),
        ("extents", "<i4"),
        ("session_error", "<i2"),
        ("regular", "S1"),
        ("hkey_un0", "S1"),
        ("dim", "<i2", (8,)),
        ("vox_units", "S4"),
        ("cal_units", "S8"),
        ("unused1", "<i2"),
        ("datatype", "<i2"),
        ("bitpix", "<i2"),
        ("dim_un0", "<i2"),
        ("pixdim", "<f4", (8,)),
        ("vox_offset", "<f4"),
        ("funused1", "<f4"),
        ("funused2", "<f4"),
        ("funused3", "<f4"),
        ("cal_max", "<f4"),
        ("cal_min", "<f4"),
        ("compressed", "<i4"),
        ("verified", "<i4"),
        ("glmax", "<i4"),
        ("glmin", "<i4"),
        ("descrip", "S80"),
        ("aux_file", "S24"),
        ("orient", "S1"),
        ("originator", "S10"),
        ("generated", "S10"),
        ("scannum", "S10"),
        ("patient_id", "S10"),
        ("exp_date", "S10"),
        ("exp_time", "S10"),
        ("hist_un0", "S3"),
        ("views", "<i4"),
        ("vols_added", "<i4"),
        ("start_field", "<i4"),
        ("field_skip", "<i4"),
        ("omax", "<i4"),
        ("omin", "<i4"),
        ("smax", "<i4"),
        ("smin", "<i4"),
    ]
)
"""Analyze 7.5 header. Assumes little-endian byte order for numbers"""


def read_hdr(p: _Path) -> _Arr[HDR_NUMPY_DTYPE]:
    """helper function for reading header"""
    with open(p, "rb") as f:
        x = _np.ndarray(shape=(), dtype=HDR_NUMPY_DTYPE, buffer=f.read())
    return x


def read_analyze75(
    p: _Path,
    transpose: _Opt[_Tup[int, ...]] = (3, 1, 0, 4),
    squeeze=True,
) -> _Tup[_Arr[_Any], _Arr[HDR_NUMPY_DTYPE]]:
    """
    Decode analyze 7.5 images into numpy array and a numpy record for a header.
    :param p: path to image. suffixes `.img`, `.hdr` are
    added using `pathlib.Path.with_suffix`.
    :param transpose: if not None, transpose.
    by default, analyze7.5 has (w, h, c, ptr) axes order.
    (3, 1, 0, 4) permutes this to (c, h, w, ptr).
    :param squeeze: use `np.squeeze`
    :return: image, metadata
    """
    meta = read_hdr(p.with_suffix(".hdr"))
    dtype = DATA_TYPE_CODES[int(meta["datatype"])]
    if type(dtype) == str:
        raise TypeError(f"Unsupported data type {dtype}")
    dim = meta["dim"]
    shape = dim[1 : dim[0] + 1]
    x = _np.fromfile(p.with_suffix(".img"), dtype=dtype).reshape(shape)
    x = x if transpose is None else x.transpose(transpose)
    x = _np.squeeze(x) if squeeze else x
    return x, meta
