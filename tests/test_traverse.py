"""Parity for the batched kernels, the BVH descent, and the whole pipeline.

The strongest statement these make is against brute force: for every ray, the
closest hit found through the R-tree and the closest hit found by testing every
triangle have to be the same triangle at the same distance. Anything the tree
prunes too aggressively shows up immediately as a disagreement.
"""

import numpy as np
import pytest

import mojortreray as mrr
from mojortreray import _lib, reference

# Two coplanar triangles meeting along a shared edge are hit at the same place
# to within a rounding error, so "the closest hit" is a band, not a point.
TIE = 1e-11


def brute_force_winners(rays, soup, ntri, tmin=0.0, tmax=np.inf):
    """For each ray, the minimum hit distance and the primitives that reach it.

    A scene of quads split into two triangles has coplanar pairs, and a ray
    through the shared diagonal hits both at the same place to within a rounding
    error -- the two evaluations differ in the last bit or two. Which one a
    traversal reports is a tie-break, not an error, so the contract under test
    is that the reported primitive is inside the tolerance band around the
    minimum, not that it equals one particular member of a set.
    """
    n = len(rays)
    offsets = np.arange(n + 1, dtype=np.int64) * ntri
    t, _ = reference.ray_triangle_intersect_batch(
        rays, np.tile(np.arange(ntri, dtype=np.int32), n), offsets, soup, tmin, tmax
    )
    out = []
    for r in range(n):
        seg = t[offsets[r] : offsets[r + 1]]
        hit = seg >= 0.0
        if not hit.any():
            out.append((-1, set()))
            continue
        best = seg[hit].min()
        out.append((best, set(np.nonzero(hit & (seg <= best * (1.0 + TIE)))[0].tolist())))
    return out


def assert_closest(rays, soup, ntri, t, prim, tmin=0.0, tmax=np.inf):
    winners = brute_force_winners(rays, soup, ntri, tmin, tmax)
    unique = 0
    hits = 0
    for r, (best, prims) in enumerate(winners):
        if best < 0:
            assert t[r] == pytest.approx(reference.MISS)
            assert prim[r] < 0
            continue
        hits += 1
        assert t[r] == pytest.approx(best, rel=TIE, abs=TIE)
        assert int(prim[r]) in prims, (r, prim[r], sorted(prims))
        unique += len(prims) == 1
    # A tie-tolerant comparison is vacuous if every ray were a tie; the
    # unrestricted case above is where the low tie rate is checked.
    assert hits > 0 and unique >= 1


def pair_arrays(scene, rays, tmin=0.0, tmax=np.inf):
    pair_tri, offsets = scene.candidate_pairs(rays, tmin, tmax)
    return np.ascontiguousarray(pair_tri), np.ascontiguousarray(offsets)


def mojo_triangle_batch(rays, pair_tri, offsets, soup, tmin, tmax, cull, eps, workers):
    t = np.full(len(pair_tri), reference.MISS)
    hit = np.zeros(len(pair_tri))
    _lib.lib().mrr_ray_triangle_intersect_batch(
        rays.ctypes.data, pair_tri.ctypes.data, offsets.ctypes.data, soup.ctypes.data,
        tmin, tmax, int(cull), eps,
        t.ctypes.data, hit.ctypes.data, len(rays), workers,
    )
    return t, hit


def mojo_aabb_batch(rays, boxes, box_idx, offsets, tmin, tmax, eps, workers):
    t = np.full(len(box_idx), reference.MISS)
    hit = np.zeros(len(box_idx))
    _lib.lib().mrr_ray_aabb_intersect_batch(
        rays.ctypes.data, boxes.ctypes.data, box_idx.ctypes.data, offsets.ctypes.data,
        tmin, tmax, eps, t.ctypes.data, hit.ctypes.data, len(rays), workers,
    )
    return t, hit


# ------------------------------------------------------------ batch pair
def test_batch_intersect_matches_the_reference(scene, rays):
    pair_tri, offsets = pair_arrays(scene, rays)
    assert len(pair_tri) > 0
    t, hit = mojo_triangle_batch(
        rays, pair_tri, offsets, scene.soup, 0.0, np.inf, False, 1e-9, 1
    )
    want_t, want_hit = reference.ray_triangle_intersect_batch(
        rays, pair_tri, offsets, scene.soup, 0.0, np.inf
    )
    assert np.array_equal(hit, want_hit)
    assert np.array_equal(hit != 0, t >= 0.0)
    assert np.allclose(np.where(hit != 0, t, 0.0), np.where(want_hit != 0, want_t, 0.0))


def test_batch_and_worker_counts_agree(scene, rays):
    pair_tri, offsets = pair_arrays(scene, rays)
    out = [
        mojo_triangle_batch(
            rays, pair_tri, offsets, scene.soup, 0.0, np.inf, False, 1e-9, w
        )
        for w in (1, 2, 4)
    ]
    for t, hit in out[1:]:
        assert np.array_equal(t, out[0][0])
        assert np.array_equal(hit, out[0][1])


def test_segment_reductions_match_the_reference(scene, rays):
    pair_tri, offsets = pair_arrays(scene, rays)
    t, hit = reference.ray_triangle_intersect_batch(
        rays, pair_tri, offsets, scene.soup, 0.0, np.inf
    )
    n = len(rays)

    got_t = np.zeros(n)
    got_p = np.zeros(n)
    _lib.lib().mrr_segment_closest_hit(
        offsets.ctypes.data, t.ctypes.data, hit.ctypes.data,
        got_t.ctypes.data, got_p.ctypes.data, n,
    )
    want_t, want_p = reference.segment_closest_hit(offsets, t, hit)
    assert np.allclose(got_t, want_t)
    assert np.array_equal(got_p, want_p)

    flag = np.zeros(n, np.int32)
    _lib.lib().mrr_segment_any_hit(
        offsets.ctypes.data, hit.ctypes.data, flag.ctypes.data, n
    )
    assert np.array_equal(flag != 0, reference.segment_any_hit(offsets, hit))

    counts = np.zeros(n + 1, np.int32)
    _lib.lib().mrr_segment_hit_count(
        offsets.ctypes.data, hit.ctypes.data, counts.ctypes.data, n
    )
    raw = np.array(counts)
    assert np.array_equal(raw, reference.segment_hit_count(offsets, hit))

    total = _lib.lib().mrr_segment_prefix_sum(counts.ctypes.data, n)
    assert total == int(counts[-1]) == int(raw.sum())
    assert np.array_equal(counts, reference.segment_prefix_sum(raw))

    ray_out = np.zeros(max(total, 1), np.int32)
    t_out = np.zeros(max(total, 1))
    prim_out = np.zeros(max(total, 1))
    _lib.lib().mrr_segment_scatter(
        offsets.ctypes.data, counts.ctypes.data, t.ctypes.data, hit.ctypes.data,
        ray_out.ctypes.data, t_out.ctypes.data, prim_out.ctypes.data, n,
    )
    want = reference.segment_scatter(offsets, counts, t, hit)
    assert np.array_equal(ray_out[:total], want[0])
    assert np.allclose(t_out[:total], want[1])
    assert np.array_equal(prim_out[:total], want[2])
    assert np.all(t_out[:total] >= 0.0)


def test_aabb_batch_matches_the_reference(rays):
    rng = np.random.default_rng(11)
    a = rng.uniform(-2, 6, (40, 3))
    b = rng.uniform(-2, 6, (40, 3))
    boxes = np.concatenate([np.minimum(a, b), np.maximum(a, b)], axis=1)
    box_idx = rng.integers(0, 40, len(rays) * 3, dtype=np.int32)
    offsets = (np.arange(len(rays) + 1) * 3).astype(np.int32)
    t, hit = mojo_aabb_batch(rays, boxes, box_idx, offsets, 0.0, np.inf, 0.0, 4)
    want_t, want_hit = reference.ray_aabb_intersect_batch(
        rays, boxes, box_idx, offsets, 0.0, np.inf, 0.0
    )
    assert np.array_equal(hit, want_hit)
    assert np.array_equal(hit != 0, t >= 0.0)
    assert np.allclose(np.where(hit != 0, t, 0.0), np.where(want_hit != 0, want_t, 0.0))


# -------------------------------------------------------------- the BVH
def test_bvh_build_has_two_children_and_one_run_per_leaf(scene):
    bvh = scene.bvh()
    assert bvh.ntris == scene.ntriangles
    counts = bvh.nint[:, 3]
    assert np.all(counts[counts > 0] <= 4)
    for node in np.nonzero(counts == 0)[0]:
        assert bvh.nint[node, 0] > 0 and bvh.nint[node, 1] > 0
    assert np.array_equal(np.sort(bvh.tris), np.arange(scene.ntriangles))
    for node in range(bvh.nboxes):
        c, first = int(bvh.nint[node, 3]), int(bvh.nint[node, 2])
        if c:
            assert first + c <= bvh.ntris


def test_bvh_bounds_contain_their_primitives(scene):
    bvh = scene.bvh()
    root_lo, root_hi = bvh.bounds
    lo, hi = scene.lo[bvh.tris], scene.hi[bvh.tris]
    assert np.all(lo >= root_lo - 1e-12) and np.all(hi <= root_hi + 1e-12)


def test_bvh_traversal_matches_brute_force(scene, rays):
    bvh = scene.bvh()
    t, prim, _, _ = mrr.traverse_bvh(rays, bvh, scene.soup)
    assert_closest(rays, scene.soup, scene.ntriangles, t, prim)
    assert np.all(t[prim >= 0] >= 0.0)


def test_bvh_traversal_agrees_with_the_rtree_path(scene, rays):
    bvh = scene.bvh()
    t_bvh, prim_bvh, _, _ = mrr.traverse_bvh(rays, bvh, scene.soup)
    hits = mrr.intersect_rays(rays, scene)
    t_rtree = np.array([h.t if h else reference.MISS for h in hits])
    prim_rtree = np.array([h.prim if h else -1.0 for h in hits])
    assert np.allclose(t_bvh, t_rtree, rtol=TIE, atol=TIE)
    # Both structures report a hit in the same band, so the reported primitive
    # has to be one of the triangles that reach it.
    assert_closest(rays, scene.soup, scene.ntriangles, t_bvh, prim_bvh)


def test_bvh_visit_order_starts_at_the_root(scene, rays):
    bvh = scene.bvh()
    _, _, order, ocount = mrr.traverse_bvh(rays, bvh, scene.soup)
    stride = mrr.bvh.ORDER_STRIDE
    seen = 0
    for i in range(len(rays)):
        visited = order[i * stride : i * stride + ocount[i]]
        assert ocount[i] >= 1
        assert visited[0] == 0
        assert np.all(visited < bvh.nboxes)
        seen += len(visited)
    assert seen > len(rays)


def test_bvh_occlusion_agrees_with_intersection(scene, rays):
    bvh = scene.bvh()
    blocked = mrr.occluded_bvh(rays, bvh, scene.soup)
    _, prim, _, _ = mrr.traverse_bvh(rays, bvh, scene.soup)
    assert np.array_equal(blocked, prim >= 0)


def test_bvh_traversal_under_the_numpy_backend_agrees(scene, rays):
    bvh = scene.bvh()
    mrr.set_backend("mojo")
    t_m, prim_m, _, _ = mrr.traverse_bvh(rays, bvh, scene.soup)
    mrr.set_backend("numpy")
    t_n, prim_n, _, _ = mrr.traverse_bvh(rays, bvh, scene.soup)
    mrr.set_backend("mojo")
    assert np.allclose(t_m, t_n, rtol=TIE, atol=TIE)
    for r, (best, prims) in enumerate(
        brute_force_winners(rays, scene.soup, scene.ntriangles)
    ):
        if best < 0:
            assert prim_m[r] < 0 and prim_n[r] < 0
            continue
        assert int(prim_m[r]) in prims and int(prim_n[r]) in prims


def test_bvh_save_and_load_round_trips(scene, tmp_path):
    bvh = scene.bvh()
    path = str(tmp_path / "bvh.npz")
    bvh.save(path)
    again = mrr.BVH.load(path)
    assert np.array_equal(again.nbox, bvh.nbox)
    assert np.array_equal(again.nint, bvh.nint)
    assert np.array_equal(again.tris, bvh.tris)


def test_bvh_respects_tmax(scene, rays):
    bvh = scene.bvh()
    t, prim, _, _ = mrr.traverse_bvh(rays, bvh, scene.soup, 0.0, 3.0)
    assert np.all(t[prim >= 0] <= 3.0)
    assert_closest(rays, scene.soup, scene.ntriangles, t, prim, 0.0, 3.0)


def test_traverse_ranged_splits_on_the_interval(scene, rays):
    bvh = scene.bvh()
    lo = np.zeros(len(rays))
    hi = np.full(len(rays), 3.0)
    t, prim = mrr.traverse_ranged(rays, bvh, scene.soup, lo, hi)
    scalar_t, scalar_prim, _, _ = mrr.traverse_bvh(rays, bvh, scene.soup, 0.0, 3.0)
    assert np.allclose(t, scalar_t)
    assert np.array_equal(prim, scalar_prim)
    # A mixed-interval batch has to split into runs and still be exact.
    mixed = np.linspace(0.5, 3.0, len(rays))
    t2, prim2 = mrr.traverse_ranged(rays, bvh, scene.soup, mixed, np.full(len(rays), 3.0))
    for i in range(len(rays)):
        best, prims = brute_force_winners(
            rays[i : i + 1], scene.soup, scene.ntriangles, mixed[i], 3.0
        )[0]
        if best < 0:
            assert t2[i] == pytest.approx(reference.MISS)
            assert prim2[i] < 0
        else:
            assert t2[i] == pytest.approx(best, rel=TIE, abs=TIE)
            assert int(prim2[i]) in prims


# --------------------------------------------------------- the pipeline
def test_intersect_rays_matches_brute_force(scene, rays):
    hits = mrr.intersect_rays(rays, scene)
    got_t = np.array([h.t if h else reference.MISS for h in hits])
    got_p = np.array([h.prim if h else -1.0 for h in hits])
    assert_closest(rays, scene.soup, scene.ntriangles, got_t, got_p)


def test_intersect_rays_backends_agree(scene, rays):
    a = mrr.intersect_rays(rays, scene)
    mrr.set_backend("numpy")
    b = mrr.intersect_rays(rays, scene)
    mrr.set_backend("mojo")
    winners = brute_force_winners(rays, scene.soup, scene.ntriangles)
    for r, (x, y) in enumerate(zip(a, b)):
        assert (x is None) == (y is None)
        if x is None:
            continue
        assert x.t == pytest.approx(y.t, rel=TIE, abs=TIE)
        best, prims = winners[r]
        assert x.t == pytest.approx(best, rel=TIE, abs=TIE)
        assert x.prim in prims and y.prim in prims


def test_intersect_ray_uses_the_ray_interval(scene, rays):
    ray = mrr.Ray(rays[0, :3], rays[0, 3:], 0.0, 1.0)
    assert mrr.intersect_ray(ray, scene) is None
    wide = mrr.Ray(rays[0, :3], rays[0, 3:])
    hit = mrr.intersect_ray(wide, scene)
    assert hit is not None and hit.t > 1.0
    assert hit.ray == 0


def test_hit_attributes(scene, rays):
    hits = [h for h in mrr.intersect_rays(rays, scene) if h is not None]
    assert hits
    for h in hits[:20]:
        ray = h.scene.rays[h.ray]
        assert np.allclose(h.point, ray[:3] + h.t * ray[3:])
        assert np.isclose(np.linalg.norm(h.normal), 1.0)
        u, v = h.uv
        face = h.face
        assert np.allclose(
            face[0] + u * (face[1] - face[0]) + v * (face[2] - face[0]),
            h.point, atol=1e-9,
        )
        assert "Hit(" in repr(h)


def test_intersect_all_groups_by_ray_and_sorts_by_t(scene, rays):
    hits = mrr.intersect_all(rays, scene)
    assert len(hits) > 0
    assert np.all(np.diff(hits.ray) >= 0)
    for r in range(len(rays)):
        sel = hits.t[hits.ray == r]
        assert np.all(np.diff(sel) >= 0)
    first = mrr.intersect_rays(rays, scene)
    for r, h in enumerate(first):
        sel = hits.t[hits.ray == r]
        if h is None:
            assert sel.size == 0
        else:
            assert sel.size >= 1 and sel[0] == pytest.approx(h.t, rel=1e-12)
    assert len(hits.offsets) == len(rays) + 1
    assert sum(1 for _ in hits) == len(hits)
    for entry in hits:
        assert entry.t >= 0.0 and entry.prim >= 0


def test_occluded_agrees_with_intersect(scene, rays):
    hits = mrr.intersect_rays(rays, scene)
    blocked = mrr.occluded_rays(rays, scene)
    assert np.array_equal(blocked, [h is not None for h in hits])
    single = mrr.occluded(mrr.Ray(rays[0, :3], rays[0, 3:]), scene)
    assert single == bool(blocked[0])


def test_occluded_backends_agree(scene, rays):
    a = mrr.occluded_rays(rays, scene)
    mrr.set_backend("numpy")
    b = mrr.occluded_rays(rays, scene)
    mrr.set_backend("mojo")
    assert np.array_equal(a, b)


def test_backface_culling_keeps_a_subset_of_the_hits(scene, rays):
    two_sided = [h for h in mrr.intersect_rays(rays, scene) if h is not None]
    front_only = [h for h in mrr.intersect_rays(rays, scene, cull_backface=True) if h]
    assert 0 < len(front_only) <= len(two_sided)
    by_ray = {h.ray: h for h in two_sided}
    for h in front_only:
        assert h.ray in by_ray
        assert h.t >= by_ray[h.ray].t - 1e-12
    bvh = scene.bvh()
    t, prim, _, _ = mrr.traverse_bvh(rays, bvh, scene.soup, cull_backface=True)
    rtree = mrr.intersect_rays(rays, scene, cull_backface=True)
    assert np.allclose(t, np.array([h.t if h else reference.MISS for h in rtree]))
    assert np.array_equal(prim, np.array([h.prim if h else -1.0 for h in rtree]))


def test_tmax_truncates_the_query(scene, rays):
    hits = mrr.intersect_rays(rays, scene, 0.0, 3.0)
    for h in hits:
        if h is not None:
            assert h.t <= 3.0
    blocked = mrr.occluded_rays(rays, scene, 0.0, 3.0)
    for b, h in zip(blocked, mrr.intersect_rays(rays, scene, 0.0, 3.0)):
        assert b == (h is not None)


def test_node_culling_matches_the_slab_test(scene, rays):
    leaves = scene.leaves()
    assert len(leaves) > 0
    ray = mrr.Ray(rays[0, :3], rays[0, 3:])
    idx, entry = mrr.cull_ray_boxes(ray, leaves)
    keep = set(idx.tolist())
    for k in range(len(leaves)):
        direct = mrr.ray_aabb_intersect(
            ray.origin, ray.direction, leaves[k, :3], leaves[k, 3:], 0.0, np.inf
        )
        assert (k in keep) == (direct is not None)
        if k in keep:
            assert entry[k] == pytest.approx(direct)


def test_occluded_boxes_matches_the_per_box_test(rays):
    rng = np.random.default_rng(2)
    a = rng.uniform(-1, 5, (20, 3))
    b = rng.uniform(-1, 5, (20, 3))
    boxes = np.concatenate([np.minimum(a, b), np.maximum(a, b)], axis=1)
    got = mrr.occluded_boxes(rays, boxes)
    want = np.array(
        [
            any(
                mrr.ray_aabb_intersect(r[:3], r[3:], b[:3], b[3:], 0.0, np.inf)
                is not None
                for b in boxes
            )
            for r in rays
        ]
    )
    assert np.array_equal(got, want)



@pytest.mark.parametrize("tmin,tmax", [(0.0, np.inf), (0.0, 3.0), (1.0, 2.0), (0.0, 0.0)])
def test_query_boxes_matches_the_scalar_query_box(scene, rays, tmin, tmax):
    """The batched box pass is the scalar `query_box` done on all three axes at
    once, so it has to agree exactly -- including the empty answer for a ray
    that misses the scene, and the `inf * 0` axes the fixture contains."""
    mins, maxs, alive = scene.query_boxes(rays, tmin, tmax)
    for i, r in enumerate(rays):
        box = scene.query_box(r[:3], r[3:], tmin, tmax)
        if box is None:
            assert not alive[i]
            continue
        assert alive[i]
        assert np.array_equal(mins[i], box[:3])
        assert np.array_equal(maxs[i], box[3:])


@pytest.mark.parametrize("tmin,tmax", [(0.0, np.inf), (0.0, 3.0), (0.5, 1.5)])
def test_candidate_pairs_matches_the_per_ray_candidates(scene, rays, tmin, tmax):
    """`candidate_pairs` issues one bulk `intersection_v` instead of one
    `intersection` generator per ray, so the flattened id list, the segment
    offsets, and the order inside each segment all have to be unchanged."""
    scene.index()
    want = [scene.candidates(mrr.Ray(r[:3], r[3:], tmin, tmax)) for r in rays]
    got, offsets = scene.candidate_pairs(rays, tmin, tmax)
    assert got.dtype == np.int32
    assert offsets.dtype == np.int32
    assert len(offsets) == len(rays) + 1
    assert offsets[0] == 0
    for i, c in enumerate(want):
        seg = got[offsets[i] : offsets[i + 1]]
        assert np.array_equal(seg, c)
    assert offsets[-1] == len(got)


def test_candidate_pairs_handles_an_empty_batch(scene):
    got, offsets = scene.candidate_pairs(np.zeros((0, 6)))
    assert got.size == 0
    assert np.array_equal(offsets, np.zeros(1, np.int32))


def test_candidate_pairs_of_a_batch_that_all_miss(scene):
    """A batch of rays pointing away from the scene issues no query at all."""
    away = np.tile(np.array([100.0, 100.0, 100.0, 0.0, 0.0, 1.0]), (4, 1))
    got, offsets = scene.candidate_pairs(away)
    assert got.size == 0
    assert np.array_equal(offsets, np.zeros(5, np.int32))


def test_candidate_pairs_of_a_mixed_batch_keep_the_segments_aligned(scene, rays):
    """Dead rays own no pairs, and a live ray after a dead one still lands on
    its own segment -- the bulk query is only issued for the live rays."""
    miss = rays[-4]
    mixed = np.concatenate([rays[:1], miss[None], rays[10:11], miss[None]], axis=0)
    want = [scene.candidates(mrr.Ray(r[:3], r[3:], 0.0, np.inf)) for r in mixed]
    assert want[1].size == 0 and want[3].size == 0
    assert want[0].size > 0 and want[2].size > 0
    got, offsets = scene.candidate_pairs(mixed)
    for i, c in enumerate(want):
        assert np.array_equal(got[offsets[i] : offsets[i + 1]], c)


def test_empty_and_single_triangle_edges():
    empty = mrr.Scene(np.zeros((3, 3)), np.array([[0, 1, 2]]))
    assert empty.ntriangles == 1
    # A zero-area triangle is degenerate, so nothing blocks the ray through it.
    assert not mrr.occluded_rays(np.array([[0.0, 0, 3, 0, 0, -1]]), empty)[0]
    real = mrr.Scene(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]])
    )
    assert mrr.occluded_rays(np.array([[0.25, 0.25, 3.0, 0, 0, -1]]), real)[0]
    assert not mrr.occluded_rays(np.array([[5.0, 5.0, 3.0, 0, 0, -1]]), real)[0]


def test_rejects_faces_outside_the_vertex_array():
    with pytest.raises(ValueError):
        mrr.Scene(np.zeros((3, 3)), np.array([[0, 1, 5]]))


def test_a_transposed_ray_batch_answers_the_same_rays(scene, rays):
    """The kernels address ray `i` at element `6 * i`, so a strided view has
    to be packed before it crosses the ABI rather than read as laid out."""
    packed = np.ascontiguousarray(rays[:12])
    strided = np.asfortranarray(packed)
    assert not strided.flags.c_contiguous
    a = mrr.intersect_rays(packed, scene)
    b = mrr.intersect_rays(strided, scene)
    assert [None if h is None else h.t for h in a] == [
        None if h is None else h.t for h in b
    ]
    ta, pa, _, _ = mrr.traverse_bvh(packed, scene.bvh(), scene.soup)
    tb, pb, _, _ = mrr.traverse_bvh(strided, scene.bvh(), scene.soup)
    assert np.array_equal(ta, tb) and np.array_equal(pa, pb)
    assert np.array_equal(mrr.occluded_rays(packed, scene), mrr.occluded_rays(strided, scene))


def test_hits_offsets_agree_with_the_ray_ids(scene, rays):
    subset = rays[:20]
    hits = mrr.intersect_all(subset, scene)
    assert len(hits) == len(hits.ray)
    off = hits.offsets
    assert off[0] == 0 and off[-1] == len(hits)
    # The hits are grouped by ray, so the offsets really are the boundaries.
    assert np.all(np.diff(off) >= 0)
    for i in range(len(subset)):
        assert np.all(hits.ray[off[i] : off[i + 1]] == i)
    assert np.all(np.diff(hits.t[off[0] : off[1]]) >= 0) if off[1] > off[0] else True


def test_a_hit_keeps_the_batch_it_came_from(scene, rays):
    """`Hit.point` and `.uv` resolve against the ray the hit was found for,
    not against whatever the scene was last queried with."""
    first = [h for h in mrr.intersect_rays(rays[:1], scene) if h is not None]
    assert first
    want_point = first[0].point
    want_uv = first[0].uv
    mrr.intersect_rays(rays[40:60], scene)
    assert np.array_equal(first[0].point, want_point)
    assert first[0].uv == want_uv


# ------------------------------------------- the hoisted slab reciprocals
def aabb_batch_rows(rays, boxes, box_idx, offsets, workers=1):
    t = np.full(len(box_idx), reference.MISS)
    hit = np.zeros(len(box_idx))
    _lib.lib().mrr_ray_aabb_intersect_batch(
        rays.ctypes.data, boxes.ctypes.data, box_idx.ctypes.data, offsets.ctypes.data,
        0.0, np.inf, 0.0, t.ctypes.data, hit.ctypes.data, len(rays), workers,
    )
    return t, hit


def test_aabb_batch_agrees_with_the_single_box_test():
    """The batch builds one `SlabRay` per ray and reuses it across every box,
    so hoisting `1.0 / direction[a]` out of the per-box loop has to be exactly
    equivalent to running the single-box test per box."""
    rng = np.random.default_rng(5)
    a = rng.uniform(-2, 2, (12, 3))
    b = a + rng.uniform(0.1, 2.0, (12, 3))
    boxes = np.ascontiguousarray(np.concatenate([a, b], axis=1))
    rays = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, -5.0, 0.0, 0.0, 0.0, 1.0],
            [3.0, 3.0, 3.0, -1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # fully degenerate direction
        ],
        dtype=np.float64,
    )
    k = len(boxes)
    for ri in range(len(rays)):
        r = np.ascontiguousarray(rays[ri : ri + 1])
        idx = np.tile(np.arange(k, dtype=np.int32), 1)
        off = np.array([0, k], dtype=np.int32)
        got_t, got_hit = aabb_batch_rows(r, boxes, idx, off)
        for j in range(k):
            want = mrr.ray_aabb_intersect(
                rays[ri, :3], rays[ri, 3:], boxes[j, :3], boxes[j, 3:], 0.0, np.inf, 0.0
            )
            if want is None:
                assert got_hit[j] == 0.0
                assert got_t[j] == reference.MISS
            else:
                assert got_hit[j] == 1.0
                assert got_t[j] == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_aabb_batch_axis_parallel_ray_matches_the_reference():
    """A ray travelling exactly along an axis exercises the `par_*` branch of
    the hoisted slab test -- the case where `1.0 / 0.0` would be a NaN."""
    rng = np.random.default_rng(6)
    a = rng.uniform(-1, 1, (30, 3))
    b = a + rng.uniform(0.1, 1.5, (30, 3))
    boxes = np.ascontiguousarray(np.concatenate([a, b], axis=1))
    rays = np.asarray(
        [
            [0.0, 0.0, -5.0, 0.0, 0.0, 1.0],
            [-5.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    box_idx = np.tile(np.arange(len(boxes), dtype=np.int32), len(rays))
    offsets = (np.arange(len(rays) + 1, dtype=np.int32) * len(boxes)).astype(np.int32)
    for workers in (1, 2):
        t, hit = aabb_batch_rows(rays, boxes, box_idx, offsets, workers)
        want_t, want_hit = reference.ray_aabb_intersect_batch(
            rays, boxes, box_idx, offsets, 0.0, np.inf, 0.0
        )
        assert np.array_equal(hit, want_hit)
        assert np.array_equal(hit != 0, t >= 0.0)
        assert np.allclose(
            np.where(hit != 0, t, 0.0), np.where(want_hit != 0, want_t, 0.0)
        )


def test_bvh_axis_parallel_rays_match_the_reference(scene):
    """The BVH interior test uses the hoisted slab ray too; axis-parallel rays
    must still find exactly what brute force finds."""
    tri = np.arange(scene.ntriangles)
    rays = np.asarray(
        [
            [0.5, 0.5, -5.0, 0.0, 0.0, 1.0],
            [-5.0, 0.5, 0.5, 1.0, 0.0, 0.0],
            [0.5, 5.0, 0.5, 0.0, -1.0, 0.0],
            [5.0, 5.0, 5.0, 0.0, 0.0, 0.0],  # degenerate direction
        ],
        dtype=np.float64,
    )
    bvh = scene.bvh()
    t, prim, _, _ = mrr.traverse_bvh(rays, bvh, scene.soup)
    want_t, _ = reference.brute_force_intersect(rays, scene.soup, tri, 0.0, np.inf)
    for i in range(len(rays)):
        if want_t[i] < 0.0:
            assert prim[i] < 0
            assert t[i] == reference.MISS
        else:
            assert prim[i] >= 0
            assert t[i] == pytest.approx(want_t[i], rel=1e-9, abs=1e-9)
