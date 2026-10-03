"""Ray/scene queries.

This is the API a caller actually uses. Each function does the same three
stages in the same order:

1. the R-tree turns each ray into a candidate list (`Scene.candidate_pairs`);
2. the Mojo kernels test every `(ray, candidate)` pair and reduce each ray's
   segment to a single answer;
3. the surviving answers are turned back into triangle ids and, on demand,
   points and normals.

`Hit` is lazy. A million primary rays should not cost a million `np.cross`
calls, so the point, normal and barycentrics are computed when first read and
then cached on the instance.
"""

from __future__ import annotations

import os

import numpy as np

from . import bvh as _bvh
from . import primitive, reference
from ._lib import addr, f64, lib, use_mojo
from .ray import Ray, as_flat

MISS = reference.MISS


class Hit:
    """One ray/primitive intersection."""

    __slots__ = ("t", "prim", "ray", "scene", "rays", "_point", "_normal", "_uv")

    def __init__(self, t, prim, ray, scene, rays=None):
        self.t = float(t)
        self.prim = int(prim)
        self.ray = int(ray)
        self.scene = scene
        # The batch this hit was taken from, not whatever `scene.rays` holds
        # now: a `Hit` kept across a second query must still resolve its own
        # ray's `point` and `uv`.
        self.rays = scene.rays if rays is None else rays
        self._point = None
        self._normal = None
        self._uv = None

    @property
    def face(self):
        """The triangle's three vertex positions."""
        return self.scene.vertices[self.scene.faces[self.prim]]

    @property
    def uv(self):
        """Barycentric `(u, v)`, in the triangle's own parameterisation."""
        if self._uv is None:
            ray = self.rays[self.ray]
            v0, v1, v2 = self.face
            self._uv = primitive.ray_triangle_barycentric(
                v0, v1, v2, ray[:3], ray[3:]
            )
        return self._uv

    @property
    def point(self) -> np.ndarray:
        """The world-space hit point."""
        if self._point is None:
            ray = self.rays[self.ray]
            self._point = primitive.ray_at(ray[:3], ray[3:], self.t)
        return self._point

    @property
    def normal(self) -> np.ndarray:
        """The unit geometric normal, in the winding the mesh declares."""
        if self._normal is None:
            v0, v1, v2 = self.face
            n = np.cross(v1 - v0, v2 - v0)
            self._normal = n / np.sqrt(np.dot(n, n))
        return self._normal

    def __repr__(self) -> str:
        return f"Hit(ray={self.ray}, prim={self.prim}, t={self.t!r})"


class Hits:
    """Every hit of every ray, grouped by ray and sorted by `t` within a ray."""

    __slots__ = ("ray", "t", "prim", "scene", "rays")

    def __init__(self, ray, t, prim, scene, rays=None):
        self.ray = ray
        self.t = t
        self.prim = prim
        self.scene = scene
        self.rays = scene.rays if rays is None else rays

    def __len__(self) -> int:
        return len(self.t)

    @property
    def offsets(self) -> np.ndarray:
        """`(n + 1,)` slice boundaries, one per ray."""
        return np.searchsorted(self.ray, np.arange(len(self.rays) + 1))

    def __iter__(self):
        for r, t, p in zip(self.ray, self.t, self.prim):
            yield Hit(t, p, r, self.scene, self.rays)


def _run_batch(scene, rays, tmin, tmax, cull_backface, eps, workers):
    """Stages 1 and 2, returning `(t_out, pair_out)` per ray."""
    r = as_flat(rays)
    n = len(r)
    pair_tri, offsets = scene.candidate_pairs(r, tmin, tmax)
    npairs = len(pair_tri)
    t_pair = np.full(npairs, MISS)
    hit_pair = np.zeros(npairs)
    if npairs:
        v = scene.soup
        if use_mojo():
            lib().mrr_ray_triangle_intersect_batch(
                addr(r),
                addr(pair_tri),
                addr(offsets),
                addr(v),
                float(tmin),
                float(tmax),
                int(bool(cull_backface)),
                float(eps),
                addr(t_pair),
                addr(hit_pair),
                n,
                workers,
            )
        else:
            t_pair, hit_pair = reference.ray_triangle_intersect_batch(
                r, pair_tri, offsets, v, tmin, tmax, cull_backface, eps
            )
    t_out = np.full(n, MISS)
    pair_out = np.full(n, -1.0)
    if n:
        if use_mojo():
            lib().mrr_segment_closest_hit(
                addr(offsets), addr(t_pair), addr(hit_pair), addr(t_out), addr(pair_out), n
            )
        else:
            t_out, pair_out = reference.segment_closest_hit(offsets, t_pair, hit_pair)
    # `pair_out` indexes the flat pair list, which is empty when no ray had a
    # candidate at all, so the mapping is done under an explicit mask.
    prim_out = np.full(len(pair_out), -1, dtype=np.int64)
    live = pair_out >= 0
    if len(pair_tri):
        prim_out[live] = pair_tri[pair_out[live].astype(np.int64)]
    return t_out, prim_out


def intersect_rays(rays, scene, tmin=0.0, tmax=np.inf, cull_backface: bool = False, eps: float = 1e-9):
    """The closest hit of every ray, as a list of `Hit` or `None`."""
    r = as_flat(rays)
    scene.rays = r
    t_out, prim_out = _run_batch(
        scene, r, tmin, tmax, cull_backface, eps, os.cpu_count() or 1
    )
    return [
        Hit(t_out[i], prim_out[i], i, scene, r) if prim_out[i] >= 0 else None
        for i in range(len(r))
    ]


def intersect_ray(ray: Ray, scene, cull_backface: bool = False, eps: float = 1e-9):
    """The closest hit of one ray, or `None`."""
    hits = intersect_rays(
        ray.flat().reshape(1, 6), scene, ray.tmin, ray.tmax, cull_backface, eps
    )
    return hits[0]


def intersect_all(rays, scene, tmin=0.0, tmax=np.inf, cull_backface: bool = False, eps: float = 1e-9):
    """Every hit of every ray, sorted by ray and then by `t`."""
    r = as_flat(rays)
    scene.rays = r
    n = len(r)
    workers = os.cpu_count() or 1
    pair_tri, offsets = scene.candidate_pairs(r, tmin, tmax)
    npairs = len(pair_tri)
    t_pair = np.full(npairs, MISS)
    hit_pair = np.zeros(npairs)
    if npairs:
        v = scene.soup
        if use_mojo():
            lib().mrr_ray_triangle_intersect_batch(
                addr(r), addr(pair_tri), addr(offsets), addr(v),
                float(tmin), float(tmax), int(bool(cull_backface)), float(eps),
                addr(t_pair), addr(hit_pair), n, workers,
            )
        else:
            t_pair, hit_pair = reference.ray_triangle_intersect_batch(
                r, pair_tri, offsets, v, tmin, tmax, cull_backface, eps
            )
    count = np.zeros(n + 1, dtype=np.int32)
    if n:
        if use_mojo():
            lib().mrr_segment_hit_count(addr(offsets), addr(hit_pair), addr(count), n)
        else:
            count = reference.segment_hit_count(offsets, hit_pair).astype(np.int32)
    if use_mojo():
        total = lib().mrr_segment_prefix_sum(addr(count), n)
    else:
        count = reference.segment_prefix_sum(count)
        total = int(count[-1])
    ray_out = np.zeros(max(total, 1), dtype=np.int32)
    t_out = np.zeros(max(total, 1))
    prim_out = np.zeros(max(total, 1))
    if n:
        if use_mojo():
            lib().mrr_segment_scatter(
                addr(offsets), addr(count), addr(t_pair), addr(hit_pair),
                addr(ray_out), addr(t_out), addr(prim_out), n,
            )
        else:
            ray_out, t_out, prim_out = reference.segment_scatter(
                offsets, count, t_pair, hit_pair
            )
    keep = int(total)
    ray_out, t_out, prim_out = ray_out[:keep], t_out[:keep], prim_out[:keep]
    order = np.lexsort((t_out, ray_out))
    return Hits(ray_out[order], t_out[order], pair_tri[prim_out[order].astype(np.int64)], scene, r)


def occluded_rays(rays, scene, tmin=0.0, tmax=np.inf, cull_backface: bool = False, eps: float = 1e-9):
    """Whether each ray is blocked by anything, without recording what."""
    r = as_flat(rays)
    workers = os.cpu_count() or 1
    pair_tri, offsets = scene.candidate_pairs(r, tmin, tmax)
    npairs = len(pair_tri)
    n = len(r)
    if not npairs:
        return np.zeros(n, dtype=bool)
    v = scene.soup
    t_pair = np.full(npairs, MISS)
    hit_pair = np.zeros(npairs)
    if use_mojo():
        lib().mrr_ray_triangle_intersect_batch(
            addr(r), addr(pair_tri), addr(offsets), addr(v),
            float(tmin), float(tmax), int(bool(cull_backface)), float(eps),
            addr(t_pair), addr(hit_pair), n, workers,
        )
    else:
        t_pair, hit_pair = reference.ray_triangle_intersect_batch(
            r, pair_tri, offsets, v, tmin, tmax, cull_backface, eps
        )
    flag = np.zeros(max(n, 1), dtype=np.int32)
    if n:
        if use_mojo():
            lib().mrr_segment_any_hit(addr(offsets), addr(hit_pair), addr(flag), n)
        else:
            flag = reference.segment_any_hit(offsets, hit_pair).astype(np.int32)
    return flag != 0


def occluded(ray: Ray, scene, cull_backface: bool = False, eps: float = 1e-9) -> bool:
    """Whether one ray is blocked by anything."""
    return bool(
        occluded_rays(
            ray.flat().reshape(1, 6), scene, ray.tmin, ray.tmax, cull_backface, eps
        )[0]
    )


def cull_ray_boxes(ray: Ray, boxes, tmin: float = 0.0, tmax: float = np.inf, eps: float = 0.0):
    """Which of `boxes` the ray's box overlaps, with the entry parameter.

    `boxes` is `(k, 6)` of `lo.xyz hi.xyz`. This is the node test a traversal
    performs, exposed on its own: it is what makes the R-tree's answer worth
    having, and what a BVH does to every interior node.
    """
    b = f64(boxes).reshape(-1, 6)
    k = len(b)
    idx = np.arange(k, dtype=np.int32)
    offsets = np.array([0, k], dtype=np.int32)
    rays = ray.flat().reshape(1, 6)
    t = np.full(k, MISS)
    hit = np.zeros(k)
    if k:
        if use_mojo():
            lib().mrr_ray_aabb_intersect_batch(
                addr(rays), addr(b), addr(idx), addr(offsets),
                float(tmin), float(tmax), float(eps),
                addr(t), addr(hit), 1, 1,
            )
        else:
            t, hit = reference.ray_aabb_intersect_batch(
                rays, b, idx, offsets, tmin, tmax, eps
            )
    return np.nonzero(hit != 0.0)[0].astype(np.int32), t


def occluded_boxes(rays, boxes, tmin=0.0, tmax=np.inf, eps: float = 0.0):
    """Whether each ray hits any box in a flat `(k, 6)` box array.

    The shadow-ray query against a scene of boxes: no triangles, no tree, just
    the slab test over every `(ray, box)` pair.
    """
    r = as_flat(rays)
    b = f64(boxes).reshape(-1, 6)
    n, k = len(r), len(b)
    blocked = np.zeros(n, dtype=bool)
    if not n or not k:
        return blocked
    pair_box = np.tile(np.arange(k, dtype=np.int32), n)
    offsets = (np.arange(n + 1, dtype=np.int32) * k).astype(np.int32)
    t = np.full(n * k, MISS)
    hit = np.zeros(n * k)
    if use_mojo():
        lib().mrr_ray_aabb_intersect_batch(
            addr(r), addr(b), addr(pair_box), addr(offsets),
            float(tmin), float(tmax), float(eps),
            addr(t), addr(hit), n, os.cpu_count() or 1,
        )
    else:
        t, hit = reference.ray_aabb_intersect_batch(
            r, b, pair_box, offsets, tmin, tmax, eps
        )
    if use_mojo():
        flag = np.zeros(n, dtype=np.int32)
        lib().mrr_segment_any_hit(addr(offsets), addr(hit), addr(flag), n)
        return flag != 0
    return reference.segment_any_hit(offsets, hit)


def traverse_bvh(rays, bvh, vertices, tmin=0.0, tmax=np.inf, cull_backface: bool = False, eps: float = 1e-9):
    """Closest hit per ray from a BVH, bypassing the R-tree entirely."""
    return _bvh.traverse(rays, bvh, vertices, tmin, tmax, cull_backface, eps)


def occluded_bvh(rays, bvh, vertices, tmin=0.0, tmax=np.inf, cull_backface: bool = False, eps: float = 1e-9):
    """Occlusion from a BVH, bypassing the R-tree entirely."""
    blocked, _ = _bvh.any_hit(rays, bvh, vertices, tmin, tmax, cull_backface, eps)
    return blocked
