"""NumPy reference implementation.

There is no package on PyPI or conda-forge called `rtree-ray`, so this file is
what mojo-rtree-ray is compared against. Every routine here is a direct
transcription of the same branch structure the Mojo kernels use, written the way
the routine is normally written in NumPy, so a disagreement between the two is a
real disagreement and not a difference of formulation.

The batched routines are vectorised over the flat pair list, which is the
strongest baseline a NumPy implementation can offer: it is memory-bandwidth
bound and multithreaded through NumPy's own expression machinery.
"""

from __future__ import annotations

import numpy as np

MISS = -1.0
FAR = 1.7976931348623157e308


# ------------------------------------------------------------------- ray
def ray_make(origin, direction):
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.sqrt(np.dot(d, d))
    return np.concatenate([np.asarray(origin, dtype=np.float64), d])


def ray_at(origin, direction, t):
    return np.asarray(origin, dtype=np.float64) + t * np.asarray(
        direction, dtype=np.float64
    )


def ray_offset(origin, direction, t):
    return ray_at(origin, direction, t)


def ray_transform(ray, m):
    ray = np.asarray(ray, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64).reshape(4, 4)
    o = m[:3, :3] @ ray[:3] + m[:3, 3]
    d = m[:3, :3] @ ray[3:]
    return np.concatenate([o, d])


# ------------------------------------------------------------ primitive
def _slab(origin, direction, lo, hi, tmin, tmax, eps):
    """Vector slab test. `origin` and `lo`/`hi` are `(n, 3)`."""
    t0 = np.full(len(origin), tmin, dtype=np.float64)
    t1 = np.full(len(origin), tmax, dtype=np.float64)
    ok = np.ones(len(origin), dtype=bool)
    for a in range(3):
        d = direction[:, a]
        o = origin[:, a]
        zero = d == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            ta = (lo[:, a] - o) * inv
            tb = (hi[:, a] - o) * inv
        near, far = ta, tb
        swap = near > far
        ta = np.where(swap, far, near)
        tb = np.where(swap, near, far)
        ta = np.where(zero, -np.inf, ta)
        tb = np.where(zero, np.inf, tb)
        t0 = np.maximum(t0, ta)
        t1 = np.minimum(t1, tb)
        ok &= np.where(zero, (o >= lo[:, a]) & (o <= hi[:, a]), t0 <= t1 + eps)
    return np.where(ok & (t1 >= 0.0), t0, MISS), ok & (t1 >= 0.0)


def ray_aabb_intersect(origin, direction, lo, hi, tmin=0.0, tmax=np.inf, eps=0.0):
    o = np.asarray(origin, dtype=np.float64).reshape(1, 3)
    d = np.asarray(direction, dtype=np.float64).reshape(1, 3)
    a = np.asarray(lo, dtype=np.float64).reshape(1, 3)
    b = np.asarray(hi, dtype=np.float64).reshape(1, 3)
    t, ok = _slab(o, d, a, b, tmin, tmax, eps)
    return float(t[0]) if ok[0] else None


def ray_aabb_any_hit(origin, direction, lo, hi, tmin=0.0, tmax=np.inf):
    o = np.asarray(origin, dtype=np.float64).reshape(1, 3)
    d = np.asarray(direction, dtype=np.float64).reshape(1, 3)
    a = np.asarray(lo, dtype=np.float64).reshape(1, 3)
    b = np.asarray(hi, dtype=np.float64).reshape(1, 3)
    _, ok = _slab(o, d, a, b, tmin, tmax, 0.0)
    return bool(ok[0])


def ray_sphere_intersect(origin, direction, center, radius, tmin=0.0, tmax=np.inf):
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    oc = o - c
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(oc, d))
    cc = float(np.dot(oc, oc)) - radius * radius
    disc = b * b - 4.0 * a * cc
    if disc < 0.0:
        return None
    sq = np.sqrt(disc)
    for t in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
        if t >= tmin and t <= tmax:
            return float(t)
    return None


def _moller_trumbore(v0, v1, v2, origin, direction, tmin, tmax, cull_backface, eps):
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    if cull_backface:
        ok = det > 0.0
    else:
        ok = ~((det > -eps) & (det < eps))
    # A degenerate determinant divides to an infinity and then to a NaN, and a
    # NaN fails every comparison -- which is exactly how the Mojo kernel rejects
    # the same pair. The warnings are the same non-event.
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_det = 1.0 / det
        tvec = origin - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        qvec = np.cross(tvec, e1)
        v = np.einsum("ij,ij->i", direction, qvec) * inv_det
        t = np.einsum("ij,ij->i", e2, qvec) * inv_det
        inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0)
    ok &= inside
    ok &= (t >= tmin) & (t <= tmax)
    return np.where(ok, t, MISS), ok


def ray_triangle_intersect(
    v0, v1, v2, origin, direction, tmin=0.0, tmax=np.inf, cull_backface=False, eps=1e-9
):
    t, ok = _moller_trumbore(
        np.asarray(v0, dtype=np.float64).reshape(1, 3),
        np.asarray(v1, dtype=np.float64).reshape(1, 3),
        np.asarray(v2, dtype=np.float64).reshape(1, 3),
        np.asarray(origin, dtype=np.float64).reshape(1, 3),
        np.asarray(direction, dtype=np.float64).reshape(1, 3),
        tmin,
        tmax,
        cull_backface,
        eps,
    )
    return float(t[0]) if ok[0] else None


def ray_triangle_barycentric(v0, v1, v2, origin, direction, eps=1e-9):
    e1 = np.asarray(v1, np.float64) - np.asarray(v0, np.float64)
    e2 = np.asarray(v2, np.float64) - np.asarray(v0, np.float64)
    d = np.asarray(direction, np.float64)
    pvec = np.cross(d, e2)
    det = float(np.dot(e1, pvec))
    if det > -eps and det < eps:
        return 0.0, 0.0
    inv_det = 1.0 / det
    tvec = np.asarray(origin, np.float64) - np.asarray(v0, np.float64)
    u = float(np.dot(tvec, pvec)) * inv_det
    v = float(np.dot(d, np.cross(tvec, e1))) * inv_det
    return u, v


def triangle_bounds(vertices, tri):
    vertices = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(tri, dtype=np.int64)
    v = vertices[tri]
    return v.min(axis=1), v.max(axis=1)


def box_surface_area(lo, hi):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    e = hi - lo
    return 2.0 * (e[..., 0] * e[..., 1] + e[..., 1] * e[..., 2] + e[..., 2] * e[..., 0])


# ------------------------------------------------------------- traverse
def _ray_of_pair(offsets):
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def ray_triangle_intersect_batch(
    rays, tri, offsets, vertices, tmin, tmax, cull_backface=False, eps=1e-9
):
    """`(rays, tri, offsets, soup, tmin, tmax)` -> `(t, hit)` per pair.

    `soup` is the flat `(ntri, 9)` triangle array the kernels read, so a
    candidate is one row of it.
    """
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 6)
    tri = np.asarray(tri, dtype=np.int64)
    soup = np.asarray(vertices, dtype=np.float64).reshape(-1, 9)
    offsets = np.asarray(offsets, dtype=np.int64)
    if len(tri) == 0:
        empty = np.zeros(0)
        return empty, empty
    which = _ray_of_pair(offsets)
    origin = rays[which, :3]
    direction = rays[which, 3:]
    v = soup[tri].reshape(-1, 3, 3)
    t, ok = _moller_trumbore(
        v[:, 0], v[:, 1], v[:, 2], origin, direction, tmin, tmax, cull_backface, eps
    )
    return t, ok.astype(np.float64)


def ray_aabb_intersect_batch(rays, boxes, box_idx, offsets, tmin, tmax, eps=0.0):
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 6)
    # The kernels read boxes as flat `(k, 6)` of `lo.xyz hi.xyz`, which is the
    # same memory as `(k, 2, 3)`.
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 2, 3)
    box_idx = np.asarray(box_idx, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    if len(box_idx) == 0:
        empty = np.zeros(0)
        return empty, empty
    which = _ray_of_pair(offsets)
    b = boxes[box_idx]
    t, ok = _slab(
        rays[which, :3], rays[which, 3:], b[:, 0], b[:, 1], tmin, tmax, eps
    )
    return t, ok.astype(np.float64)


def segment_closest_hit(offsets, t, hit):
    """`(t, prim)` per ray. `prim` is the pair index, or -1."""
    offsets = np.asarray(offsets, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    hit = np.asarray(hit, dtype=np.float64) != 0.0
    n = len(offsets) - 1
    out_t = np.full(n, MISS)
    out_p = np.full(n, -1.0)
    for r in range(n):
        a, b = offsets[r], offsets[r + 1]
        sel = np.nonzero(hit[a:b])[0]
        if sel.size:
            j = sel[np.argmin(t[a:b][sel])]
            out_t[r] = t[a + j]
            out_p[r] = a + j
    return out_t, out_p


def segment_any_hit(offsets, hit):
    offsets = np.asarray(offsets, dtype=np.int64)
    hit = np.asarray(hit, dtype=np.float64) != 0.0
    n = len(offsets) - 1
    out = np.zeros(n, dtype=bool)
    for r in range(n):
        out[r] = bool(hit[offsets[r] : offsets[r + 1]].any())
    return out


def segment_hit_count(offsets, hit):
    offsets = np.asarray(offsets, dtype=np.int64)
    hit = np.asarray(hit, dtype=np.float64) != 0.0
    n = len(offsets) - 1
    out = np.zeros(n + 1, dtype=np.int32)
    for r in range(n):
        out[r] = hit[offsets[r] : offsets[r + 1]].sum()
    return out


def segment_prefix_sum(count):
    count = np.asarray(count, dtype=np.int32)
    out = np.zeros_like(count)
    out[1:] = np.cumsum(count[:-1])
    return out


def segment_scatter(offsets, slots, t, hit):
    """Returns `(ray, t, prim)` grouped by ray, in pair order within a ray."""
    offsets = np.asarray(offsets, dtype=np.int64)
    slots = np.asarray(slots, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    hit = np.asarray(hit, dtype=np.float64) != 0.0
    n = len(offsets) - 1
    ray_out = []
    t_out = []
    prim_out = []
    for r in range(n):
        a, b = offsets[r], offsets[r + 1]
        w = slots[r]
        for k in np.nonzero(hit[a:b])[0] + a:
            ray_out.append(r)
            t_out.append(t[k])
            prim_out.append(k)
            w += 1
    return (
        np.asarray(ray_out, dtype=np.int32),
        np.asarray(t_out, dtype=np.float64),
        np.asarray(prim_out, dtype=np.float64),
    )


def brute_force_intersect(rays, vertices, tri, tmin=0.0, tmax=np.inf, cull_backface=False):
    """Every ray against every triangle. The honest baseline for an
    accelerated traversal: it has no structure to get wrong."""
    rays = np.asarray(rays, dtype=np.float64).reshape(-1, 6)
    tri = np.asarray(tri, dtype=np.int64)
    n = rays.shape[0]
    if n == 0:
        return np.zeros(0), np.full(0, -1.0)
    offsets = np.arange(n + 1, dtype=np.int64) * len(tri)
    pair_tri = np.tile(tri, n)
    t, _ = ray_triangle_intersect_batch(
        rays, pair_tri, offsets, vertices, tmin, tmax, cull_backface
    )
    hit_t, pair = segment_closest_hit(offsets, t, (t >= 0.0).astype(np.float64))
    # `segment_closest_hit` reports the pair index; here pairs are the triangle
    # list tiled once per ray, so the primitive id is the pair index modulo it.
    ntri = len(tri)
    prim = np.where(pair >= 0, pair % ntri, -1.0)
    return hit_t, prim
