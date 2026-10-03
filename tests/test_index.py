"""`Index` against the real `rtree.index.Index`, and `Scene`'s culling against
brute force.

The `rtree` package is a genuine install of the spatial index this project
builds on, so it is the thing to compare against here: every method of `Index`
has to return exactly what libspatialindex returns for the same operations.

The other half is the contract the traversal depends on. `Scene.candidates` may
return anything *extra* -- an over-approximation only costs time -- but it may
never omit a triangle that actually intersects the ray, or a hit silently
disappears. That is asserted directly.
"""

import os

import numpy as np
import pytest
import rtree

import mojortreray as mrr
from mojortreray import reference


def boxes_2d(seed=0, n=200):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 10, (n, 2))
    b = a + rng.uniform(0.05, 2.0, (n, 2))
    return a, b


def boxes_3d(seed=0, n=200):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 10, (n, 3))
    b = a + rng.uniform(0.05, 2.0, (n, 3))
    return a, b


def coords(lo, hi):
    return [tuple(np.concatenate([l, h])) for l, h in zip(lo, hi)]


def test_index_matches_rtree_for_2d():
    lo, hi = boxes_2d()
    ours = mrr.Index()
    theirs = rtree.index.Index()
    for i, c in enumerate(coords(lo, hi)):
        ours.insert(i, c)
        theirs.insert(i, c)
    assert ours.get_size() == len(theirs) == 200
    assert len(ours) == len(theirs)
    assert np.allclose(ours.bounds, theirs.bounds)
    for _ in range(40):
        q = np.concatenate([rng_box(2) for _ in range(1)])
        lo2, hi2 = q[:2], q[2:]
        c = tuple(np.concatenate([lo2, hi2]))
        assert sorted(ours.intersection(c)) == sorted(theirs.intersection(c))
        assert ours.count(c) == theirs.count(c)
        assert sorted(ours.nearest((lo2 + hi2) / 2, 5)) == sorted(
            theirs.nearest((lo2 + hi2) / 2, 5)
        )
        assert len(list(ours.contains(c, 0))) == len(list(theirs.contains(c, 0)))


def rng_box(dim):
    a = np.random.default_rng(abs(hash(dim)) % 9973).uniform(0, 10, dim)
    return np.concatenate([a, a + 1.0])


def test_index_matches_rtree_for_3d():
    lo, hi = boxes_3d(seed=3)
    props = rtree.index.Property()
    props.dimension = 3
    ours = mrr.Index(properties=props)
    theirs = rtree.index.Index(properties=props)
    for i, c in enumerate(coords(lo, hi)):
        ours.insert(i, c)
        theirs.insert(i, c)
    assert ours.get_size() == len(theirs)
    assert np.allclose(ours.bounds, theirs.bounds)
    q = np.concatenate([rng_box(3)])
    c = tuple(q)
    assert sorted(ours.intersection(c)) == sorted(theirs.intersection(c))
    assert ours.count(c) == theirs.count(c)
    assert sorted(ours.leaves()[0][1]) == sorted(theirs.leaves()[0][1])


def test_index_objects_and_delete_match_rtree():
    lo, hi = boxes_2d(seed=1)
    ours = mrr.Index()
    theirs = rtree.index.Index()
    for i, c in enumerate(coords(lo, hi)):
        ours.insert(i, c, obj=f"obj{i}")
        theirs.insert(i, c, obj=f"obj{i}")
    q = tuple(np.concatenate([np.array([2.0, 2.0]), np.array([6.0, 6.0])]))
    def payloads(index):
        return sorted((item.id, item.object) for item in index.intersection(q, objects=True))

    assert payloads(ours) == payloads(theirs)
    for i in range(0, 20):
        ours.delete(i, coords(lo, hi)[i])
        theirs.delete(i, coords(lo, hi)[i])
    assert ours.get_size() == len(theirs)
    assert sorted(ours.intersection(q)) == sorted(theirs.intersection(q))
    theirs.delete(20, coords(lo, hi)[20])
    ours.delete(20, coords(lo, hi)[20])
    assert sorted(ours.intersection(q)) == sorted(theirs.intersection(q))


def test_index_items_constructor_matches_insert_loop():
    items = [(i, c) for i, c in enumerate(coords(*boxes_2d(seed=4)))]
    a = mrr.Index(items=items)
    b = mrr.Index()
    for i, c in items:
        b.insert(i, c)
    assert a.get_size() == b.get_size() == len(items)
    q = tuple(np.concatenate([np.array([1.0, 1.0]), np.array([9.0, 9.0])]))
    assert sorted(a.intersection(q)) == sorted(b.intersection(q))


def test_index_rejects_an_unknown_backend():
    with pytest.raises(ValueError):
        mrr.set_backend("cuda")


def test_scene_index_holds_every_triangle(scene):
    idx = scene.index()
    assert idx.get_size() == scene.ntriangles
    lo, hi = scene.bounds
    assert np.allclose(idx.bounds[:3], lo)
    assert np.allclose(idx.bounds[3:], hi)
    assert scene.leaves().shape[1] == 6


def test_scene_candidates_are_a_superset_of_the_true_hits(scene, rays):
    """The property the whole culling stage rests on: over-selecting is
    allowed, under-selecting loses hits."""
    pair_tri, offsets = scene.candidate_pairs(rays)
    tri = np.arange(scene.ntriangles, dtype=np.int32)
    all_t, _ = reference.ray_triangle_intersect_batch(
        rays, np.tile(tri, len(rays)), (np.arange(len(rays) + 1) * scene.ntriangles),
        scene.soup, 0.0, np.inf,
    )
    for r in range(len(rays)):
        offered = set(pair_tri[offsets[r] : offsets[r + 1]].tolist())
        seg = all_t[r * scene.ntriangles : (r + 1) * scene.ntriangles]
        for k in np.nonzero(seg >= 0.0)[0]:
            assert int(k) in offered, (r, int(k))


def test_scene_query_box_is_well_formed_when_the_ray_reaches_the_scene(scene, rays):
    lo, hi = scene.bounds
    checked = 0
    for r in rays:
        if not mrr.ray_aabb_any_hit(r[:3], r[3:], lo, hi, 0.0, np.inf):
            continue
        box = scene.query_box(r[:3], r[3:], 0.0, np.inf)
        assert np.all(box[:3] <= box[3:])
        checked += 1
    assert checked > 0


def test_scene_bvh_covers_every_triangle(scene):
    bvh = scene.bvh()
    assert np.array_equal(np.sort(bvh.tris), np.arange(scene.ntriangles))
    assert len(scene) == scene.ntriangles


def test_a_compiler_that_cannot_run_raises_builderror(tmp_path):
    """The import is lazy, so the first kernel call is where a missing compiler
    surfaces. It has to be a `BuildError` carrying the command, not a raw
    `FileNotFoundError` out of `subprocess`, which says nothing useful."""
    from mojortreray import _lib

    original_lib = _lib.LIB
    original_mojo = os.environ.get("MOJORTRE_RAY_MOJO")
    _lib.LIB = str(tmp_path / "out.so")
    os.environ["MOJORTRE_RAY_MOJO"] = str(tmp_path / "no-such-mojo")
    try:
        with pytest.raises(_lib.BuildError):
            _lib.build(force=True)
    finally:
        _lib.LIB = original_lib
        if original_mojo is None:
            os.environ.pop("MOJORTRE_RAY_MOJO", None)
        else:
            os.environ["MOJORTRE_RAY_MOJO"] = original_mojo
