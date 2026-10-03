"""The `Ray` value type.

Field names, argument order and defaults follow the de-facto standard for this
problem -- `origin`, `direction`, `tmin`, `tmax`, with `at`, `offset`,
`normalize` and `transform` -- so a caller does not have to learn a new API.

The flat 6-double form is the on-the-wire layout: it is what the Mojo kernels
read, so a batch of rays is a contiguous `(n, 6)` array and no marshalling
happens between here and the kernel.
"""

from __future__ import annotations

import numpy as np

from . import primitive


class Ray:
    __slots__ = ("origin", "direction", "tmin", "tmax")

    def __init__(self, origin, direction, tmin: float = 0.0, tmax: float = np.inf):
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.direction = np.asarray(direction, dtype=np.float64).reshape(3)
        self.tmin = float(tmin)
        self.tmax = float(tmax)

    def at(self, t: float) -> np.ndarray:
        """The point at arc length `t`."""
        return primitive.ray_at(self.origin, self.direction, t)

    def offset(self, t: float) -> np.ndarray:
        """`at(t)`, promoted to a fresh origin."""
        return primitive.ray_offset(self.origin, self.direction, t)

    def normalize(self) -> "Ray":
        """Scale the direction to unit length in place."""
        flat = primitive.ray_make(self.origin, self.direction)
        self.direction = flat[3:]
        return self

    def transform(self, m) -> "Ray":
        """A new ray in the frame `m` maps into (a row-major 4x4)."""
        flat = primitive.ray_transform(self.flat(), m)
        return Ray(flat[:3], flat[3:], self.tmin, self.tmax)

    def flat(self) -> np.ndarray:
        """The `(6,)` layout the kernels read."""
        return np.concatenate([self.origin, self.direction])

    @property
    def invdir(self) -> np.ndarray:
        """Reciprocal direction, with infinities for the zero components -- the
        form the slab test is usually written against."""
        with np.errstate(divide="ignore"):
            return 1.0 / self.direction

    def __repr__(self) -> str:
        return (
            f"Ray(origin={self.origin.tolist()}, "
            f"direction={self.direction.tolist()}, "
            f"tmin={self.tmin!r}, tmax={self.tmax!r})"
        )


def as_flat(rays) -> np.ndarray:
    """Coerce a `Ray`, a `(6,)` array or an `(n, 6)` array to `(n, 6)`.

    The result is always C-contiguous: the kernels address ray `i` at
    element `6 * i`, so a transposed or otherwise strided view would be read
    with the wrong stride and silently answer with another ray's result.
    """
    if isinstance(rays, Ray):
        return rays.flat().reshape(1, 6)
    return np.ascontiguousarray(np.asarray(rays, dtype=np.float64)).reshape(-1, 6)
