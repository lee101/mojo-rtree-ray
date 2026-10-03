"""mojo-rtree-ray against the NumPy reference on the same data.

There is no `rtree-ray` package to benchmark against -- the name is not on PyPI
and not on conda-forge -- so the baseline is `mojortreray.reference`: the same
algorithms written the way they are normally written in NumPy, vectorised over
the flat pair list and running through NumPy's own multithreaded expression
machinery. That is a strong baseline, and where the Mojo kernels lose to it the
table says so.

The R-tree stage is timed separately and separately labelled, because neither
side owns it: both go through libspatialindex via the `rtree` package, and the
difference in those rows is the cost of the wrapper, not of any kernel.

The end-to-end rows are timed on a small scene on purpose. A camera ray's
bounding box over an unbounded interval covers a large part of a dense scene,
so the R-tree hands back a large candidate list and those rows measure the cull
stage more than the kernel. The BVH rows are the accelerated path a renderer
would actually take.

Run with `pixi run bench`. Do not run this file directly -- the pixi task holds
a machine-wide lock so that concurrent jobs cannot distort the numbers.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojortreray as mrr  # noqa: E402
from mojortreray import _lib, reference  # noqa: E402

WORKERS = os.cpu_count() or 1


def timeit(fn, repeat: int = 5) -> float:
    """Best-of-N. The minimum is the least contaminated estimate of the work
    itself; a mean here measures whatever else the machine was doing."""
    fn()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def machine() -> str:
    model = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (
        f"{platform.system()} {platform.machine()}, {WORKERS} logical CPUs, "
        f"{model or 'unknown CPU'}"
    )


def camera_rays(n, lo, hi, seed=0):
    """A pinhole spread over the scene, the way a renderer would issue them."""
    rng = np.random.default_rng(seed)
    span = hi - lo
    px = rng.uniform(0.1, 0.9, n) * span[0] + lo[0]
    py = rng.uniform(0.1, 0.9, n) * span[1] + lo[1]
    origin = np.column_stack([np.full(n, lo[0] - 0.05 * span[0]), py, px])
    target = np.column_stack(
        [
            rng.uniform(0, 1, n) * span[0] + lo[0],
            rng.uniform(0, 1, n) * span[1] + lo[1],
            rng.uniform(0, 1, n) * span[2] + lo[2],
        ]
    )
    direction = target - origin
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return np.ascontiguousarray(np.concatenate([origin, direction], axis=1))


def mojo_triangle_batch(rays, pair_tri, offsets, soup):
    t = np.full(len(pair_tri), reference.MISS)
    hit = np.zeros(len(pair_tri))
    _lib.lib().mrr_ray_triangle_intersect_batch(
        rays.ctypes.data,
        pair_tri.ctypes.data,
        offsets.ctypes.data,
        soup.ctypes.data,
        0.0,
        np.inf,
        0,
        1e-9,
        t.ctypes.data,
        hit.ctypes.data,
        len(rays),
        WORKERS,
    )
    return t, hit


def mojo_aabb_batch(rays, boxes, box_idx, offsets):
    t = np.full(len(box_idx), reference.MISS)
    hit = np.zeros(len(box_idx))
    _lib.lib().mrr_ray_aabb_intersect_batch(
        rays.ctypes.data,
        boxes.ctypes.data,
        box_idx.ctypes.data,
        offsets.ctypes.data,
        0.0,
        np.inf,
        0.0,
        t.ctypes.data,
        hit.ctypes.data,
        len(rays),
        WORKERS,
    )
    return t, hit


CASES = []


def case(name, note="", baseline=True, repeat=5):
    def deco(fn):
        CASES.append((name, note, fn, baseline, repeat))
        return fn

    return deco


SCENE = mrr.Scene.grid(shape=(32, 32, 32), seed=1)
SCENE_LO, SCENE_HI = SCENE.bounds
SMALL = mrr.Scene.grid(shape=(16, 16, 16), seed=2)
SMALL_LO, SMALL_HI = SMALL.bounds
TINY = mrr.Scene.grid(shape=(12, 12, 12), seed=3)
TINY_LO, TINY_HI = TINY.bounds


@case("ray_triangle_intersect_batch (4M pairs, 50k rays x 80 candidates)")
def _():
    rays = camera_rays(50_000, SCENE_LO, SCENE_HI)
    width = 80
    pair_tri = np.tile(np.arange(width, dtype=np.int32), 50_000)
    offsets = (np.arange(50_001, dtype=np.int32) * width).astype(np.int32)
    return (
        lambda: mojo_triangle_batch(rays, pair_tri, offsets, SCENE.soup),
        lambda: reference.ray_triangle_intersect_batch(
            rays, pair_tri, offsets, SCENE.soup, 0.0, np.inf
        ),
    )


@case("ray_aabb_intersect_batch (4M pairs, 20k rays x 200 nodes)")
def _():
    rng = np.random.default_rng(2)
    a = rng.uniform(-5, 5, (200, 3))
    b = a + rng.uniform(0.1, 3.0, (200, 3))
    boxes = np.ascontiguousarray(np.concatenate([a, b], axis=1))
    width = 200
    box_idx = np.tile(np.arange(width, dtype=np.int32), 20_000)
    offsets = (np.arange(20_001, dtype=np.int32) * width).astype(np.int32)
    origin = rng.uniform(-5, 5, (20_000, 3))
    direction = rng.normal(size=(20_000, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    rays = np.ascontiguousarray(np.concatenate([origin, direction], axis=1))
    return (
        lambda: mojo_aabb_batch(rays, boxes, box_idx, offsets),
        lambda: reference.ray_aabb_intersect_batch(
            rays, boxes, box_idx, offsets, 0.0, np.inf, 0.0
        ),
    )


@case("BVH closest-hit vs NumPy brute force (3k rays, 8192 triangles)", repeat=3)
def _():
    rays = camera_rays(3_000, SMALL_LO, SMALL_HI)
    bvh = SMALL.bvh()
    soup = SMALL.soup
    tri = np.arange(SMALL.ntriangles)
    return (
        lambda: mrr.traverse_bvh(rays, bvh, soup),
        lambda: reference.brute_force_intersect(rays, soup, tri, 0.0, np.inf),
    )


@case("BVH occlusion vs NumPy brute force (3k rays, 8192 triangles)", repeat=3)
def _():
    rays = camera_rays(3_000, SMALL_LO, SMALL_HI)
    bvh = SMALL.bvh()
    soup = SMALL.soup
    tri = np.arange(SMALL.ntriangles)
    return (
        lambda: mrr.occluded_bvh(rays, bvh, soup),
        lambda: reference.brute_force_intersect(rays, soup, tri, 0.0, np.inf),
    )


@case(
    "BVH closest-hit throughput (200k rays, 65536 triangles)",
    note="brute force would be 13G pairs",
    baseline=False,
)
def _():
    rays = camera_rays(200_000, SCENE_LO, SCENE_HI)
    bvh = SCENE.bvh()
    soup = SCENE.soup
    return (lambda: mrr.traverse_bvh(rays, bvh, soup), None)


@case("intersect_rays end to end (3k rays, 3456 triangles)", repeat=3)
def _():
    rays = camera_rays(3_000, TINY_LO, TINY_HI)
    TINY.index()

    def numpy_path():
        mrr.set_backend("numpy")
        try:
            return mrr.intersect_rays(rays, TINY)
        finally:
            mrr.set_backend("mojo")

    return (lambda: mrr.intersect_rays(rays, TINY), numpy_path)


@case("intersect_all end to end (3k rays, 3456 triangles)", repeat=3)
def _():
    rays = camera_rays(3_000, TINY_LO, TINY_HI)
    TINY.index()

    def numpy_path():
        mrr.set_backend("numpy")
        try:
            return mrr.intersect_all(rays, TINY)
        finally:
            mrr.set_backend("mojo")

    return (lambda: mrr.intersect_all(rays, TINY), numpy_path)


@case(
    "R-tree cull for 3k rays, 3456 triangles",
    note="libspatialindex; this is the bound on the whole query",
    baseline=False,
    repeat=3,
)
def _():
    rays = camera_rays(3_000, TINY_LO, TINY_HI)
    TINY.index()
    pair, _ = TINY.candidate_pairs(rays)
    print(
        f"    (cull returns {len(pair) / len(rays):.0f} candidates per ray "
        f"out of {TINY.ntriangles} triangles)"
    )
    return (lambda: TINY.candidate_pairs(rays), None)


@case(
    "R-tree cull, wrapper vs rtree (2k rays, 3456 triangles)",
    note="both sides are libspatialindex",
    repeat=3,
)
def _():
    import rtree

    rays = camera_rays(2_000, TINY_LO, TINY_HI)
    ours = TINY.index()
    props = rtree.index.Property()
    props.dimension = 3
    theirs = rtree.index.Index(properties=props)
    coords = np.concatenate([TINY.lo, TINY.hi], axis=1)
    for f in range(TINY.ntriangles):
        theirs.insert(f, tuple(coords[f]))
    boxes = [tuple(TINY.query_box(r[:3], r[3:], 0.0, np.inf)) for r in rays]
    return (
        lambda: [list(ours.intersection(b)) for b in boxes],
        lambda: [list(theirs.intersection(b)) for b in boxes],
    )


def main() -> None:
    print(f"machine: {machine()}")
    print(f"numpy:   {np.__version__}   backend: {mrr.backend()}")
    print()
    rows = []
    for name, note, build, baseline, repeat in CASES:
        ours, theirs = build()
        a = timeit(ours, repeat)
        if theirs is None:
            b = math.nan
            flag = "no baseline"
        else:
            b = timeit(theirs, repeat)
            flag = "ok" if a <= b else "slower"
        rows.append((name, a, b, flag, note))
        if theirs is None:
            print(f"  {name}: {a * 1e3:.1f}ms  (no baseline)")
        else:
            print(f"  {name}: {a * 1e3:.1f}ms vs {b * 1e3:.1f}ms ({b / a:.2f}x) {flag}")

    print()
    print("| kernel | mojo-rtree-ray | numpy reference | ratio | note |")
    print("| --- | ---: | ---: | ---: | --- |")
    for name, a, b, flag, note in rows:
        if flag == "no baseline":
            label, ours_cell, ref_cell, ratio, cell = (
                name,
                f"{a * 1e3:.1f} ms",
                "not run",
                "-",
                note,
            )
        else:
            labelled = flag == "slower" and not note
            label = f"{name} -- mojo slower" if labelled else name
            ours_cell = f"{a * 1e3:.1f} ms"
            ref_cell = f"{b * 1e3:.1f} ms"
            ratio = f"{b / a:.2f}x"
            cell = note or (
                "Mojo loses to NumPy's multithreaded SIMD" if labelled else ""
            )
        print(f"| {label} | {ours_cell} | {ref_cell} | {ratio} | {cell} |")


if __name__ == "__main__":
    main()
