"""A flat, linearly laid out BVH over a primitive set.

Two arrays, and that is the whole structure. `nbox` is `(nodes, 6)` of
`lo.xyz hi.xyz`; `nint` is `(nodes, 4)` of `left, right, first, count`, where
`count > 0` marks a leaf holding `count` primitive ids starting at `first` in
`tris`, `count == 0` marks an interior node with exactly two children, and
`count < 0` marks an explicitly empty node. The last one exists because a
zeroed row reads as an interior node whose two children are itself, so a tree
over no primitives would descend into itself forever.

Nodes are emitted in pre-order and a leaf's primitives are written to `tris`
where that leaf is emitted, so every leaf owns a contiguous run and the inner
loop of the traversal is a walk over `tris` with no indirection beyond the id
itself.

The builder is a median split on the widest axis of the centroid bounds. That is
the cheapest build that still gives a usable tree; a surface-area-heuristic
sweep would be better and is not what this pass is for.
"""

from __future__ import annotations

import os

import numpy as np

from . import reference
from ._lib import addr, f64, lib, use_mojo

# Two pushes per interior node, so the stack never needs more than one entry
# per level plus the pending sibling. The traversal refuses to push past the
# stride rather than overrunning the buffer.
STACK_STRIDE = 128
ORDER_STRIDE = 256


class BVH:
    __slots__ = ("nbox", "nint", "tris", "workers")

    def __init__(self, nbox, nint, tris):
        self.nbox = np.ascontiguousarray(nbox, dtype=np.float64)
        self.nint = np.ascontiguousarray(nint, dtype=np.int32)
        self.tris = np.ascontiguousarray(tris, dtype=np.int32)
        self.workers = os.cpu_count() or 1

    @property
    def nboxes(self) -> int:
        return len(self.nbox)

    @property
    def ntris(self) -> int:
        return len(self.tris)

    @property
    def bounds(self):
        """`(lo, hi)` of the root."""
        return self.nbox[0, :3].copy(), self.nbox[0, 3:].copy()

    def __len__(self) -> int:
        return self.ntris

    def save(self, path: str) -> None:
        np.savez(path, nbox=self.nbox, nint=self.nint, tris=self.tris)

    @classmethod
    def load(cls, path: str) -> "BVH":
        # `np.load` on an `.npz` hands back a lazily-read `NpzFile` that holds an
        # open file descriptor until it is closed. Closing it here is what keeps a
        # loop of `load`/`save` from running out of descriptors.
        with np.load(path) as z:
            return cls(z["nbox"], z["nint"], z["tris"])

    @classmethod
    def build(cls, lo, hi, leaf_size: int = 4) -> "BVH":
        lo = f64(lo).reshape(-1, 3)
        hi = f64(hi).reshape(-1, 3)
        n = len(lo)
        if n == 0:
            # `count == -1` marks a node that holds neither primitives nor
            # children. An all-zero row would read as an interior node whose
            # two children are itself, and the traversal would never terminate.
            return cls(
                np.zeros((1, 6)),
                np.asarray([[0, 0, 0, -1]], dtype=np.int32),
                np.zeros(0, np.int32),
            )
        centroids = 0.5 * (lo + hi)
        # A leaf can hold a single primitive, so the tree can have up to `n`
        # leaves and hence up to `2n - 1` nodes; the arrays are sized for the
        # worst case and trimmed to what was actually used.
        nbox = np.zeros((2 * n + 1, 6))
        nint = np.zeros((2 * n + 1, 4), dtype=np.int32)
        order = np.empty(n, dtype=np.int32)
        written = 0
        used = 0

        def emit(ids: np.ndarray) -> int:
            # `written` is the running offset into `order`, so each leaf takes
            # its slice in one step; summing the leaf sizes here instead would
            # make the whole build quadratic in the primitive count.
            nonlocal written, used
            index = used
            used += 1
            box = ids
            nbox[index] = np.concatenate(
                [lo[box].min(axis=0), hi[box].max(axis=0)]
            )
            if len(ids) <= leaf_size:
                order[written : written + len(ids)] = ids
                nint[index] = (0, 0, written, len(ids))
                written += len(ids)
                return index
            spread = centroids[ids].max(axis=0) - centroids[ids].min(axis=0)
            axis = int(np.argmax(spread))
            sorted_ids = ids[np.argsort(centroids[ids, axis], kind="stable")]
            mid = len(sorted_ids) // 2
            left = emit(sorted_ids[:mid])
            right = emit(sorted_ids[mid:])
            nint[index] = (left, right, 0, 0)
            return index

        emit(np.arange(n, dtype=np.int32))
        return cls(nbox[:used], nint[:used], order[:written])


def traverse(
    rays,
    bvh: BVH,
    vertices,
    tmin=0.0,
    tmax=np.inf,
    cull_backface: bool = False,
    eps: float = 1e-9,
):
    """Closest hit per ray: `(t, prim, order, ocount)`, with `prim == -1` for a
    miss. `order` records which nodes each ray visited, in visit order, into a
    per-ray strided region of `ORDER_STRIDE` entries; `ocount[r]` is how many."""
    r = f64(rays).reshape(-1, 6)
    v = f64(vertices).reshape(-1, 9)
    n = len(r)
    t = np.full(n, reference.MISS)
    prim = np.full(n, -1.0)
    order = np.zeros(max(n, 1) * ORDER_STRIDE, dtype=np.int32)
    ocount = np.zeros(max(n, 1), dtype=np.int32)
    if not n:
        return t, prim, order[:0], ocount[:0]
    if use_mojo():
        stack = np.zeros(n * STACK_STRIDE, dtype=np.int32)
        lib().mrr_bvh_traverse(
            addr(r),
            addr(bvh.nbox),
            addr(bvh.nint),
            addr(bvh.tris),
            addr(v),
            addr(stack),
            float(tmin),
            float(tmax),
            int(bool(cull_backface)),
            float(eps),
            addr(t),
            addr(prim),
            addr(order),
            addr(ocount),
            n,
            bvh.workers,
            STACK_STRIDE,
            ORDER_STRIDE,
        )
        return t, prim, order, ocount
    t, prim = _traverse_numpy(r, bvh, v, float(tmin), float(tmax), cull_backface, eps)
    return t, prim, order, ocount


def traverse_ranged(
    rays,
    bvh: BVH,
    vertices,
    tmin,
    tmax,
    cull_backface: bool = False,
    eps: float = 1e-9,
):
    """Per-ray `[tmin, tmax]`. Split into maximal runs of equal bounds so the
    kernel is still entered with scalars; the loop is over runs, not rays."""
    r = f64(rays).reshape(-1, 6)
    lo = f64(tmin).reshape(-1)
    hi = f64(tmax).reshape(-1)
    n = len(r)
    t = np.full(n, reference.MISS)
    prim = np.full(n, -1.0)
    if not n:
        return t, prim
    pairs = np.stack([lo, hi], axis=1)
    change = np.nonzero(np.any(pairs[1:] != pairs[:-1], axis=1))[0] + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [n]])
    for a, b in zip(starts, stops):
        tt, pp, _, _ = traverse(r[a:b], bvh, vertices, lo[a], hi[a], cull_backface, eps)
        t[a:b] = tt
        prim[a:b] = pp
    return t, prim


def _traverse_numpy(rays, bvh, vertices, tmin, tmax, cull_backface, eps):
    """The same descent in Python, one ray at a time. Reachable only when the
    backend is set to `numpy`; `tests/test_parity.py` checks it against the Mojo
    traversal and against brute force."""
    n = len(rays)
    out_t = np.full(n, reference.MISS)
    out_p = np.full(n, -1.0)
    nbox, nint, tris = bvh.nbox, bvh.nint, bvh.tris
    soup = np.asarray(vertices, dtype=np.float64).reshape(-1, 9)
    for ri in range(n):
        origin = rays[ri, :3]
        direction = rays[ri, 3:]
        t = tmax
        best = -1
        stack = [0]
        while stack:
            node = stack.pop()
            if (
                reference.ray_aabb_intersect(
                    origin, direction, nbox[node, :3], nbox[node, 3:], tmin, t, eps
                )
                is None
            ):
                continue
            count = int(nint[node, 3])
            if count > 0:
                first = int(nint[node, 2])
                for k in range(count):
                    tri = int(tris[first + k])
                    v = soup[tri].reshape(3, 3)
                    th = reference.ray_triangle_intersect(
                        v[0], v[1], v[2], origin, direction, tmin, t, cull_backface, eps
                    )
                    if th is not None and th < t:
                        t = th
                        best = tri
                continue
            left, right = int(nint[node, 0]), int(nint[node, 1])
            tl = reference.ray_aabb_intersect(
                origin, direction, nbox[left, :3], nbox[left, 3:], tmin, t, eps
            )
            tr = reference.ray_aabb_intersect(
                origin, direction, nbox[right, :3], nbox[right, 3:], tmin, t, eps
            )
            if tl is not None and tr is not None:
                if tl <= tr:
                    stack.append(right)
                    stack.append(left)
                else:
                    stack.append(left)
                    stack.append(right)
            elif tl is not None:
                stack.append(left)
            elif tr is not None:
                stack.append(right)
        if best >= 0:
            out_t[ri] = t
            out_p[ri] = best
    return out_t, out_p


def any_hit(
    rays,
    bvh: BVH,
    vertices,
    tmin=0.0,
    tmax=np.inf,
    cull_backface: bool = False,
    eps: float = 1e-9,
):
    """Occlusion: `blocked` per ray, and the primitive that blocks it.

    Under the `numpy` backend there is no separate any-hit descent; the
    closest hit is taken and reported as the blocker, which is the same
    boolean answer with a possibly different primitive id.
    """
    r = f64(rays).reshape(-1, 6)
    v = f64(vertices).reshape(-1, 9)
    n = len(r)
    blocked = np.zeros(n, dtype=bool)
    prim = np.full(n, -1.0)
    if not n:
        return blocked, prim
    if use_mojo():
        stack = np.zeros(n * STACK_STRIDE, dtype=np.int32)
        flag = np.zeros(n, dtype=np.int32)
        lib().mrr_bvh_any_hit(
            addr(r),
            addr(bvh.nbox),
            addr(bvh.nint),
            addr(bvh.tris),
            addr(v),
            addr(stack),
            float(tmin),
            float(tmax),
            int(bool(cull_backface)),
            float(eps),
            addr(flag),
            addr(prim),
            n,
            bvh.workers,
            STACK_STRIDE,
        )
        return flag != 0, prim
    t, p = _traverse_numpy(r, bvh, v, tmin, tmax, cull_backface, eps)
    return p >= 0, p
