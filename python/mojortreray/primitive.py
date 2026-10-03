"""Public ray/primitive intersection API.

Argument order and defaults match the kernels in `src/mojortreray`: the
direction is never implicitly normalised, `tmin`/`tmax` bound the interval, and
a miss is `None` rather than a sentinel float.

Every routine dispatches to the Mojo kernel or to the NumPy reference in
`mojortreray.reference`, chosen by `mojortreray.set_backend`. The two are
required to agree; `tests/test_parity.py` is what holds them to it.
"""

from __future__ import annotations

import numpy as np

from . import reference
from ._lib import addr, f64, i32, lib, use_mojo


def _v3(a) -> np.ndarray:
    return f64(np.asarray(a, dtype=np.float64).reshape(3))


def ray_make(origin, direction) -> np.ndarray:
    """`Ray.normalize`: a flat 6-vector with a unit direction."""
    if not use_mojo():
        return reference.ray_make(origin, direction)
    o, d, dst = _v3(origin), _v3(direction), np.empty(6)
    lib().mrr_ray_make(addr(o), addr(d), addr(dst))
    return dst


def ray_at(origin, direction, t: float) -> np.ndarray:
    """The point at arc length `t`."""
    if not use_mojo():
        return reference.ray_at(origin, direction, t)
    o, d, dst = _v3(origin), _v3(direction), np.empty(3)
    lib().mrr_ray_at(addr(o), addr(d), float(t), addr(dst))
    return dst


def ray_offset(origin, direction, t: float) -> np.ndarray:
    """The point at arc length `t`, promoted to a fresh origin."""
    if not use_mojo():
        return reference.ray_offset(origin, direction, t)
    o, d, dst = _v3(origin), _v3(direction), np.empty(3)
    lib().mrr_ray_offset(addr(o), addr(d), float(t), addr(dst))
    return dst


def ray_transform(ray, m) -> np.ndarray:
    """Transform by a row-major 4x4: the origin picks up the translation
    column, the direction does not."""
    if not use_mojo():
        return reference.ray_transform(ray, m)
    r, mm, dst = f64(ray).reshape(6), f64(m).reshape(16), np.empty(6)
    lib().mrr_ray_transform(addr(r), addr(mm), addr(dst))
    return dst


def ray_aabb_intersect(
    origin, direction, lo, hi, tmin: float = 0.0, tmax: float = np.inf, eps: float = 0.0
):
    """Entry parameter of the ray/box overlap, or `None`."""
    if not use_mojo():
        return reference.ray_aabb_intersect(origin, direction, lo, hi, tmin, tmax, eps)
    o, d = _v3(origin), _v3(direction)
    a, b = _v3(lo), _v3(hi)
    t = lib().mrr_ray_aabb_intersect(
        addr(o), addr(d), addr(a), addr(b), float(tmin), float(tmax), float(eps)
    )
    return t if t >= 0.0 else None


def ray_aabb_any_hit(origin, direction, lo, hi, tmin: float = 0.0, tmax: float = np.inf):
    """Whether the ray overlaps the box at all, with no entry parameter."""
    if not use_mojo():
        return reference.ray_aabb_any_hit(origin, direction, lo, hi, tmin, tmax)
    o, d = _v3(origin), _v3(direction)
    a, b = _v3(lo), _v3(hi)
    return bool(lib().mrr_ray_aabb_any_hit(addr(o), addr(d), addr(a), addr(b), float(tmin), float(tmax)))


def ray_sphere_intersect(origin, direction, center, radius: float, tmin: float = 0.0, tmax: float = np.inf):
    """Entry parameter of the ray/sphere overlap, or `None`."""
    if not use_mojo():
        return reference.ray_sphere_intersect(origin, direction, center, radius, tmin, tmax)
    o, d, c = _v3(origin), _v3(direction), _v3(center)
    t = lib().mrr_ray_sphere_intersect(
        addr(o), addr(d), addr(c), float(radius), float(tmin), float(tmax)
    )
    return t if t >= 0.0 else None


def ray_triangle_intersect(
    v0,
    v1,
    v2,
    origin,
    direction,
    tmin: float = 0.0,
    tmax: float = np.inf,
    cull_backface: bool = False,
    eps: float = 1e-9,
):
    """Moller-Trumbore. Entry parameter, or `None`."""
    if not use_mojo():
        return reference.ray_triangle_intersect(
            v0, v1, v2, origin, direction, tmin, tmax, cull_backface, eps
        )
    a, b, c = _v3(v0), _v3(v1), _v3(v2)
    o, d = _v3(origin), _v3(direction)
    t = lib().mrr_ray_triangle_intersect(
        addr(a), addr(b), addr(c), addr(o), addr(d),
        float(tmin), float(tmax), int(bool(cull_backface)), float(eps),
    )
    return t if t >= 0.0 else None


def ray_triangle_barycentric(v0, v1, v2, origin, direction, eps: float = 1e-9):
    """The `(u, v)` of an accepted hit, for barycentric attributes."""
    if not use_mojo():
        return reference.ray_triangle_barycentric(v0, v1, v2, origin, direction, eps)
    a, b, c = _v3(v0), _v3(v1), _v3(v2)
    o, d = _v3(origin), _v3(direction)
    bu, bv = np.empty(1), np.empty(1)
    lib().mrr_ray_triangle_barycentric(
        addr(a), addr(b), addr(c), addr(o), addr(d), float(eps), addr(bu), addr(bv)
    )
    return float(bu[0]), float(bv[0])


def triangle_bounds(vertices, faces):
    """Per-triangle `(lo, hi)` of an indexed mesh, the shape the R-tree is
    built over. `faces` is `(ntri, 3)` of vertex indices.

    The index range is checked before the kernel is entered. `triangle_bounds`
    reads `vertices` unchecked and the only bound on the read is the int32 the
    index is narrowed to, so a face index past the end of the vertex array is a
    heap read -- a wrong answer for a small overshoot and a segmentation fault
    for a large one. This is the same check `Scene.__init__` makes, and it is
    here rather than there because this is the public entry point.
    """
    v = f64(vertices).reshape(-1, 3)
    f = i32(faces).reshape(-1, 3)
    if f.size and (f.min() < 0 or f.max() >= len(v)):
        raise ValueError(
            f"faces index outside vertices: [{int(f.min())}, {int(f.max())}] "
            f"into {len(v)} vertices"
        )
    if not use_mojo():
        return reference.triangle_bounds(v, f)
    lo, hi = np.empty((len(f), 3)), np.empty((len(f), 3))
    lib().mrr_triangle_bounds(addr(v), addr(f), addr(lo), addr(hi), len(f))
    return lo, hi


def box_surface_area(lo, hi):
    """`2(xy + yz + zx)` per box."""
    a, b = f64(lo).reshape(-1, 3), f64(hi).reshape(-1, 3)
    if not use_mojo():
        return reference.box_surface_area(a, b)
    out = np.empty(len(a))
    lib().mrr_box_surface_area(addr(a), addr(b), addr(out), len(a))
    return out
