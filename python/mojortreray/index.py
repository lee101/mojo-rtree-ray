"""The spatial index and the scene it indexes.

`Index` is a drop-in stand-in for `rtree.index.Index`: same constructor, same
method names, same signatures, same return shapes. It delegates to
libspatialindex through the `rtree` package, which is the one piece of this
pipeline that was not worth rewriting -- the R-tree is already C, and the
interesting work in a ray tracer happens after the tree has answered.

`Scene` is the part specific to traversal. It holds a triangle soup, computes
the per-triangle AABBs with the Mojo kernel, indexes them, and turns a ray into
a candidate list. That list is a *superset* of the true hits: the tree is
queried with the ray's own AABB over `[tmin, tmax]` clipped to the scene bounds,
and every candidate is then tested exactly. Over-selecting costs time;
under-selecting would be a correctness bug, so the direction of the error is
what matters.
"""

from __future__ import annotations

import itertools

import numpy as np
import rtree

from . import primitive
from ._lib import i32
from .bvh import BVH
from .ray import Ray, as_flat


class Index:
    """A spatial index with `rtree.index.Index`'s API.

    `coordinates` are `(left, bottom, [front], right, top, [back])` for a 2D or
    3D index; the dimension is taken from the first tuple inserted, as
    libspatialindex does.
    """

    def __init__(
        self,
        properties=None,
        items=None,
        id_generator=None,
        interleaved: bool = True,
        filename=None,
    ):
        kwargs = {"interleaved": interleaved}
        if filename is not None:
            kwargs["filename"] = filename
        if properties is not None:
            kwargs["properties"] = properties
        self._index = rtree.index.Index(**kwargs)
        self._ids = itertools.count()
        if items is not None:
            for item in items:
                self.insert(*item)

    def insert(self, id, coordinates, obj=None) -> None:
        """`Index.insert`: `obj` is an optional payload the queries can return."""
        if id is None:
            id = next(self._ids)
        self._index.insert(id, coordinates, obj)

    def delete(self, id, coordinates) -> None:
        """`Index.delete`.

        The coordinates are required, and that is not an omission: `rtree` 1.4.1
        made the argument mandatory, and a wrapper that accepted an older
        one-argument form would fail at the call rather than at the definition.
        """
        self._index.delete(id, coordinates)

    def intersection(self, coordinates, objects: bool = False):
        """Ids whose extent overlaps the query box, or the payload objects."""
        return self._index.intersection(coordinates, objects)

    def intersection_v(self, mins, maxs):
        """Bulk form of `intersection`: a whole `(n, d)` batch of query boxes
        in one call, returning `(ids, counts)`.

        The scalar form costs one ctypes round trip per query box and another
        per result; this is the same libspatialindex descent with the loop
        moved into C, which is what the batched kernels need anyway.
        """
        return self._index.intersection_v(mins, maxs)

    def count(self, coordinates) -> int:
        """How many entries overlap the query box."""
        return self._index.count(coordinates)

    def nearest(self, coordinates, num_results, objects: bool = False):
        """The `num_results` entries nearest to a point."""
        return self._index.nearest(coordinates, num_results, objects)

    def contains(self, coordinates, obj=None) -> bool:
        """Whether an entry at `coordinates` is present."""
        return self._index.contains(coordinates, obj)

    def leaves(self):
        """The tree's node boxes, in the tree's own order."""
        return self._index.leaves()

    def get_size(self) -> int:
        """The number of entries indexed.

        `rtree` deprecated `Index.get_size()` in favour of `__len__` but kept
        the name, so the count is taken from `len` and the name is kept here for
        callers that were written against the older spelling.
        """
        return len(self._index)

    @property
    def bounds(self):
        """The whole index's extent, as libspatialindex reports it: four
        numbers for a 2D index, six for a 3D one."""
        return self._index.bounds

    def __len__(self) -> int:
        return self._index.__len__()


class Scene:
    """A triangle soup with an R-tree over its per-triangle AABBs."""

    def __init__(self, vertices, faces, leaf_size: int = 4):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = i32(faces).reshape(-1, 3)
        if self.faces.size and (
            self.faces.min() < 0 or self.faces.max() >= len(self.vertices)
        ):
            raise ValueError("faces index outside vertices")
        self.soup = np.ascontiguousarray(self.vertices[self.faces]).reshape(-1, 9)
        self.lo, self.hi = primitive.triangle_bounds(self.vertices, self.faces)
        if not self.ntriangles:
            # `min`/`max` over an empty axis has no identity and raises, which
            # would turn an empty scene -- legal, and the one a ray that misses
            # everything is answered from -- into an opaque numpy failure at
            # construction instead of an empty scene.
            zero = np.zeros(3)
            self._bounds = (zero, zero)
        else:
            self._bounds = (self.lo.min(axis=0), self.hi.max(axis=0))
        self.leaf_size = leaf_size
        self._index: Index | None = None
        self._bvh: BVH | None = None

    @classmethod
    def grid(cls, shape=(8, 8, 8), cell: float = 1.0, jitter: float = 0.25, seed: int = 0):
        """A regular grid of jittered quads, each split into two triangles:
        enough scattered geometry to make candidate lists non-trivial, and
        coplanar triangle pairs to exercise tie-breaking, without needing a
        mesh file."""
        rng = np.random.default_rng(seed)
        base = np.indices(shape).reshape(3, -1).T * cell
        base = base + rng.uniform(-jitter, jitter, base.shape) * cell
        unit = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ) * cell
        vertices = (base[:, None, :] + unit[None, :, :]).reshape(-1, 3)
        first = 4 * np.arange(len(base), dtype=np.int32)
        faces = np.empty((2 * len(base), 3), dtype=np.int32)
        faces[0::2] = np.stack([first, first + 1, first + 2], axis=1)
        faces[1::2] = np.stack([first, first + 2, first + 3], axis=1)
        return cls(vertices, faces)

    @property
    def ntriangles(self) -> int:
        return len(self.faces)

    def __len__(self) -> int:
        return self.ntriangles

    @property
    def bounds(self):
        """`(lo, hi)` of the whole scene.

        Reduced once in `__init__`: this is read once per ray by
        `query_box`, and re-reducing `lo`/`hi` there cost more than the
        tree query it was clipping.
        """
        return self._bounds

    def index(self) -> Index:
        if self._index is None:
            props = rtree.index.Property()
            props.dimension = 3
            idx = Index(properties=props)
            coords = np.concatenate([self.lo, self.hi], axis=1)
            for f in range(self.ntriangles):
                idx.insert(f, tuple(coords[f]))
            self._index = idx
        return self._index

    def bvh(self) -> BVH:
        """The BVH over the same AABBs, built on first use."""
        if self._bvh is None:
            self._bvh = BVH.build(self.lo, self.hi, self.leaf_size)
        return self._bvh

    def leaves(self) -> np.ndarray:
        """`(k, 6)` of the R-tree's node boxes, in the tree's own order.

        libspatialindex reports leaves as `(level, ids, bbox)`; the boxes are
        what a traversal can use, so the level and id lists are dropped here.
        """
        boxes = [entry[2] for entry in self.index().leaves()]
        return np.asarray(boxes, dtype=np.float64).reshape(-1, 6)

    def query_box(self, origin, direction, tmin: float, tmax: float):
        """The box to hand the R-tree for a ray, or `None` if the ray misses the
        scene outright.

        The box is the ray's own AABB over `[tmin, tmax]`, clipped to the scene
        on every axis. The clip is also the emptiness test: two intervals that
        do not overlap produce a minimum past the maximum, and libspatialindex
        rejects that, so the ray is answered here instead.
        """
        lo, hi = self.bounds
        box = np.empty(6)
        for a in range(3):
            d = direction[a]
            o = origin[a]
            if d == 0.0:
                # A parallel ray never leaves the slab, so the whole ray is one
                # point on this axis. Computing `o + tmax * 0` instead would be
                # `inf * 0`, a NaN, and NaN silently defeats every min and max.
                near, far = o, o
            else:
                p0 = o + tmin * d
                p1 = o + tmax * d
                near = min(p0, p1)
                far = max(p0, p1)
            near = max(near, lo[a])
            far = min(far, hi[a])
            if near > far:
                return None
            box[a] = near
            box[3 + a] = far
        return box

    def query_boxes(self, rays, tmin: float, tmax: float):
        """`query_box` for a whole `(n, 6)` batch, as `(mins, maxs, alive)`.

        This is the same arithmetic as the scalar form, done on all three axes
        at once, because the batched kernels ask for every ray at once and the
        scalar loop was the per-call Python cost that dominated the cull.
        Rays whose box is empty come back with `alive` false and no query
        issued for them; they own zero pairs, exactly as `query_box`
        returning `None` implies.
        """
        r = np.asarray(rays, dtype=np.float64).reshape(-1, 6)
        lo, hi = self.bounds
        o = r[:, 0:3]
        d = r[:, 3:6]
        # `tmax` is usually infinite, and `inf * 0.0` is a NaN, which then
        # silently defeats every min and max. The parallel axes are selected
        # explicitly, as the scalar form does, and the NaN is discarded.
        with np.errstate(invalid="ignore"):
            p0 = o + tmin * d
            p1 = o + tmax * d
        par = d == 0.0
        near = np.where(par, o, np.minimum(p0, p1))
        far = np.where(par, o, np.maximum(p0, p1))
        near = np.maximum(near, lo)
        far = np.minimum(far, hi)
        alive = ~(near > far).any(axis=1)
        return (
            np.ascontiguousarray(near, dtype=np.float64),
            np.ascontiguousarray(far, dtype=np.float64),
            alive,
        )

    def candidates(self, ray: Ray) -> np.ndarray:
        """Triangle ids the R-tree offers for one ray.

        `intersection` is lazy in `rtree`; the ids are materialised here because
        the batched kernels need them all before they can start.
        """
        box = self.query_box(ray.origin, ray.direction, ray.tmin, ray.tmax)
        if box is None:
            return np.zeros(0, np.int32)
        return np.fromiter(self.index().intersection(tuple(box)), dtype=np.int32)

    def candidate_pairs(self, rays, tmin=0.0, tmax=np.inf):
        """The flattened `(pair_tri, offsets)` the batched kernels read.

        `rays` is `(n, 6)`; `offsets` has `n + 1` entries, so ray `i` owns
        `offsets[i]:offsets[i + 1]`.

        The whole batch goes through one bulk `intersection_v` call rather
        than one `intersection` generator per ray. The result is the same
        list of ids in the same order -- `intersection_v` returns the flat
        ids grouped by query box, with a per-box count -- but the per-ray
        ctypes round trip, and the `np.concatenate` of the per-ray chunks,
        are both gone. The ids arrive as int64 and are narrowed once.
        """
        r = as_flat(rays)
        n = len(r)
        counts = np.zeros(n, dtype=np.int64)
        pair_tri = np.zeros(0, np.int32)
        if n:
            mins, maxs, alive = self.query_boxes(r, tmin, tmax)
            live = np.flatnonzero(alive)
            if len(live):
                ids, got = self.index().intersection_v(
                    np.ascontiguousarray(mins[live]),
                    np.ascontiguousarray(maxs[live]),
                )
                counts[live] = got
                if len(ids):
                    pair_tri = np.array(ids, dtype=np.int32)
        offsets = np.zeros(n + 1, dtype=np.int32)
        np.cumsum(counts, out=offsets[1:], dtype=np.int32)
        return pair_tri, offsets
