import numpy as np
from numpy.typing import ArrayLike as _ArrL

from ..prelude import *

_ArrF = np.ndarray[np.dtype[float]]


class Coo2D:
    """array of (n_coordinates, 2) floats, (x, y) order, representing spatial coordinates"""

    __slots__ = ("coo2d",)

    def __init__(self, coo2d: _ArrF):
        self.coo2d: _ArrF = np.array(coo2d, dtype=float)
        assert self.coo2d.ndim == 2 and self.coo2d.shape[1] == 2

    def n_coordinates(self) -> int:
        """:return len(self.coo2d)"""
        return len(self.coo2d)

    __len__ = n_coordinates

    def move_2d(self, xy: _ArrL) -> "Coo2D":
        """:param xy: shape (2, ) or (n_coordinates, 2)"""
        return self.__class__(self.coo2d + xy)

    def x(self) -> _ArrF:
        """:return self.coo2d[:, 0]"""
        return self.coo2d[:, 0]

    def y(self) -> _ArrF:
        """:return self.coo2d[:, 1]"""
        return self.coo2d[:, 1]

    def rotate_2d(self, radians: float, center_xy: _ArrL) -> "Coo2D":
        """
        :param radians:
        :param center_xy: centerpoints of shape (2, ) or (n_coordinates, 2)
        """
        c = np.asarray(center_xy)  # TODO: can radians be array-like?
        a = np.cos(radians)
        b = np.sin(radians)
        rT = np.array(((a, b), (-b, a)))
        v = (self.coo2d - c) @ rT + c
        return self.__class__(v)

    def mass_center_2d(self, weights: Opt[_ArrL] = None) -> _ArrF:
        """:param weights: if present, should be of shape (n, )"""
        if weights is None:
            return np.mean(self.coo2d, axis=0)
        else:
            return np.sum(self.coo2d * weights, axis=0) / np.sum(weights)

    __repr__ = repr_slots


class RotRect(Coo2D):
    """
    coordinate array of shape (4, 2), representing rotated rectangle ROI on an image.
    first axis order is (top-left, bottom-left, bottom-right, top-right),
    second axis order is (x, y). y-axis is inverted (top < bottom).
    """

    @classmethod
    def new_non_rotated(
        cls,
        right: float,
        bottom: float,
        left: float = 0.0,
        top: float = 0.0,
    ) -> "RotRect":
        return cls(
            coo2d=np.array(
                ((left, top), (left, bottom), (right, bottom), (right, top)),
                dtype=float,
            )
        )

    def rotation(self) -> float:
        tl, bl, br, tr = self.coo2d
        x, y = br - tl
        a1 = np.arctan2(y, x)
        x, y = bl - tr
        a2 = np.arctan2(y, -x)
        return (a1 + a2) / 2

    def wh(self) -> Tuple[float, float]:
        tl, bl, br, tr = self.coo2d
        w = np.linalg.norm((br - bl + tr - tl) / 2)
        h = np.linalg.norm((br - tr + bl - tl) / 2)
        return w, h

    def wh_int(self) -> Tuple[int, int]:
        w, h = self.wh()
        return round(w), round(h)

    def non_rotated(self) -> "RotRect":
        w, h = self.wh_int()
        return self.new_non_rotated(h, w)
