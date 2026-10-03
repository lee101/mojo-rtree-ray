"""Parity between the Mojo kernels and the NumPy reference, per routine.

Every test here runs the same call under both backends and compares. A test
that only checked "it did not crash" would prove nothing about a port whose
entire claim is that the arithmetic came across intact.
"""

import math

import numpy as np
import pytest

import mojortreray as mrr

from .conftest import both_backends


def close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# ------------------------------------------------------------------ ray
def test_ray_make_normalises_only_when_asked():
    a, b = both_backends(mrr.ray_make, [1.0, 2.0, 3.0], [0.0, 0.0, 5.0])
    assert np.array_equal(a, b)
    assert np.allclose(a[:3], [1.0, 2.0, 3.0])
    assert np.allclose(a[3:], [0.0, 0.0, 1.0])


def test_ray_at_and_offset():
    a, b = both_backends(mrr.ray_at, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 2.5)
    assert np.allclose(a, b) and np.allclose(a, [2.5, 1.0, 0.0])
    a, b = both_backends(mrr.ray_offset, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 2.5)
    assert np.allclose(a, b) and np.allclose(a, [2.5, 1.0, 0.0])


def test_ray_transform_drops_the_translation_from_the_direction():
    m = np.eye(4)
    m[:3, 3] = [10.0, 20.0, 30.0]
    m[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    a, b = both_backends(mrr.ray_transform, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], m)
    assert np.allclose(a, b)
    assert np.allclose(a[:3], [10.0, 21.0, 30.0])
    assert np.allclose(a[3:], [0.0, 1.0, 0.0])


def test_ray_transform_keeps_length_under_a_scaling():
    m = np.eye(4)
    m[0, 0] = 3.0
    a, _ = both_backends(mrr.ray_transform, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], m)
    assert np.allclose(a[3:], [3.0, 0.0, 0.0])


# ------------------------------------------------------------- box slab
BOX_CASES = [
    # origin, direction, lo, hi
    ([0, 0, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1]),
    ([-5, 0, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1]),
    ([-5, 5, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1]),
    ([-5, 0, 0], [0, 0, 0], [-1, -1, -1], [1, 1, 1]),
    ([-5, 0, 0], [0, 0, 0], [-1, -1, -1], [1, 0, 1]),
    ([0, 0, -5], [0, 0, 1], [-1, -1, -1], [1, 1, 1]),
    ([-5, 0, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1]),
    ([2, 0, 0], [-1, 0, 0], [-1, -1, -1], [1, 1, 1]),
]


@pytest.mark.parametrize("origin,direction,lo,hi", BOX_CASES)
def test_ray_aabb_intersect_matches(origin, direction, lo, hi):
    a, b = both_backends(mrr.ray_aabb_intersect, origin, direction, lo, hi, 0.0, np.inf)
    assert close(a, b), (a, b)


@pytest.mark.parametrize("origin,direction,lo,hi", BOX_CASES)
def test_ray_aabb_any_hit_matches(origin, direction, lo, hi):
    a, b = both_backends(mrr.ray_aabb_any_hit, origin, direction, lo, hi, 0.0, np.inf)
    assert a == b


def test_aabb_from_inside_hits_at_tmin():
    t = mrr.ray_aabb_intersect([0, 0, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1], 0.0, np.inf)
    assert t == pytest.approx(0.0)


def test_aabb_respects_the_interval_not_just_tmin():
    # The box spans t in [4, 6] along this ray, so the entry is 4 whatever
    # tmin is below it, and the interval [5, 6] is still inside the box.
    assert mrr.ray_aabb_intersect([-5, 0, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1], 3.0, 4.0) == pytest.approx(4.0)
    assert mrr.ray_aabb_intersect([-5, 0, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1], 5.0, 6.0) == pytest.approx(5.0)
    # tmin past the exit is a genuine miss: the ray has left the box by then.
    assert mrr.ray_aabb_intersect([0, 0, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1], 2.0, np.inf) is None


def test_aabb_behind_the_origin_is_a_miss():
    assert mrr.ray_aabb_intersect([5, 0, 0], [1, 0, 0], [-1, -1, -1], [1, 1, 1]) is None


def test_aabb_axis_parallel_inside_the_slab_still_hits():
    # A zero direction component times an infinite reciprocal is the classic
    # way to lose a box that the ray is in fact inside.
    assert mrr.ray_aabb_intersect([-5, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, 1]) == 4.0
    assert mrr.ray_aabb_any_hit([-5, 0, 0], [1, 0, 0], [-1, 0, -1], [1, 0, 1])
    assert not mrr.ray_aabb_any_hit([-5, 2, 0], [1, 0, 0], [-1, 0, -1], [1, 0, 1])


# -------------------------------------------------------------- sphere
def test_ray_sphere_intersect_matches():
    for origin, direction, center, radius in [
        ([0, 0, -5], [0, 0, 1], [0, 0, 0], 1.0),
        ([0, 0, 0], [1, 0, 0], [0, 0, 0], 1.0),
        ([0, 0, -5], [0, 0, 2], [0, 0, 0], 1.0),
        ([0, 0, -5], [1, 0, 0], [0, 0, 0], 1.0),
    ]:
        a, b = both_backends(
            mrr.ray_sphere_intersect, origin, direction, center, radius, 0.0, np.inf
        )
        assert close(a, b), (a, b)


def test_ray_sphere_entry_and_exit():
    assert mrr.ray_sphere_intersect([0, 0, -5], [0, 0, 1], [0, 0, 0], 1.0) == pytest.approx(4.0)
    assert mrr.ray_sphere_intersect([0, 0, -5], [1, 0, 0], [0, 0, 0], 1.0) is None
    assert mrr.ray_sphere_intersect([0, 0, -5], [0, 0, 1], [0, 0, 0], 1.0, 10.0, 20.0) is None


# ------------------------------------------------------------ triangle
UNIT = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_triangle_hit_from_above_and_below():
    front = mrr.ray_triangle_intersect(*UNIT, [0.25, 0.25, -1.0], [0, 0, 1])
    back = mrr.ray_triangle_intersect(*UNIT, [0.25, 0.25, 1.0], [0, 0, -1])
    assert front == pytest.approx(1.0)
    assert back == pytest.approx(1.0)


def test_backface_culling_is_one_sided():
    # UNIT winds counter-clockwise about +z, so a ray arriving from +z sees the
    # front face and a ray arriving from -z sees the back.
    front = [0.25, 0.25, 1.0], [0, 0, -1]
    back = [0.25, 0.25, -1.0], [0, 0, 1]
    assert mrr.ray_triangle_intersect(*UNIT, *front, cull_backface=True) is not None
    assert mrr.ray_triangle_intersect(*UNIT, *back, cull_backface=True) is None
    assert mrr.ray_triangle_intersect(*UNIT, *back) is not None


def test_triangle_misses_outside_the_edges():
    for point in [(1.5, 0.25, -1.0), (-0.5, 0.25, -1.0), (0.25, 1.5, -1.0)]:
        assert mrr.ray_triangle_intersect(*UNIT, point, [0, 0, 1]) is None


def test_triangle_degenerate_is_always_a_miss():
    flat = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    a, b = both_backends(
        mrr.ray_triangle_intersect, *flat, [1.0, 0.0, -1.0], [0, 0, 1]
    )
    assert a is None and b is None


@pytest.mark.parametrize("origin,direction", [
    ([0.25, 0.25, -1.0], [0, 0, 1]),
    ([0.9, 0.05, -1.0], [0, 0, 1]),
    ([0.25, 0.25, 1.0], [0, 0, -1]),
])
def test_triangle_intersect_matches(origin, direction):
    a, b = both_backends(
        mrr.ray_triangle_intersect, *UNIT, origin, direction, 0.0, np.inf, False, 1e-9
    )
    assert close(a, b), (a, b)


def test_barycentric_reconstructs_the_hit_point():
    origin, direction = np.array([0.25, 0.25, -1.0]), np.array([0.0, 0.0, 1.0])
    u, v = mrr.ray_triangle_barycentric(*UNIT, origin, direction)
    point = UNIT[0] + u * (UNIT[1] - UNIT[0]) + v * (UNIT[2] - UNIT[0])
    assert np.allclose(point, [0.25, 0.25, 0.0])
    assert (u, v) == pytest.approx((0.25, 0.25))


def test_triangle_bounds_matches():
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0.5, 2.0]])
    f = np.array([[0, 1, 2], [1, 2, 3]])
    a, b = both_backends(mrr.triangle_bounds, v, f)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    lo, hi = b
    assert np.array_equal(lo, [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    assert np.array_equal(hi, [[1.0, 1.0, 0.0], [1.0, 1.0, 2.0]])


def test_box_surface_area_matches():
    lo = np.array([[0.0, 0, 0], [-1.0, -2.0, -3.0]])
    hi = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    a, b = both_backends(mrr.box_surface_area, lo, hi)
    assert np.array_equal(a, b)
    assert a[0] == pytest.approx(2 * (1 * 2 + 2 * 3 + 3 * 1))
    assert a[1] == pytest.approx(2 * (2 * 4 + 4 * 6 + 6 * 2))


def test_ray_class_round_trip():
    ray = mrr.Ray([0, 0, 0], [0, 0, 5], 1.0, 100.0)
    ray.normalize()
    assert np.allclose(ray.direction, [0, 0, 1])
    assert ray.tmin == 1.0 and ray.tmax == 100.0
    assert np.allclose(ray.at(3.0), [0, 0, 3.0])
    assert np.allclose(ray.offset(3.0), [0, 0, 3.0])
    moved = ray.transform(np.eye(4))
    assert np.allclose(moved.origin, ray.origin)
    assert "Ray(" in repr(ray)
    assert np.allclose(mrr.as_flat(ray).shape, (1, 6))


def test_ray_sphere_origin_on_the_surface_hits_at_tmin():
    # The near root is exactly 0, the default `tmin`. A strict `t > tmin`
    # drops it and reports the far root at 2r instead.
    a, b = both_backends(
        mrr.ray_sphere_intersect, [0, 0, 1], [0, 0, -1], [0, 0, 0], 1.0
    )
    assert a == pytest.approx(0.0)
    assert close(a, b)


def test_index_arrays_reject_non_whole_numbers():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(TypeError):
        mrr.triangle_bounds(vertices, np.array([[0, 1, 2.5]]))
    with pytest.raises(TypeError):
        mrr.Scene(vertices, np.array([[0, 1, 2.5]]))
    # A whole number in a float array is a legitimate index, not a mistake.
    lo, hi = mrr.triangle_bounds(vertices, np.array([[0.0, 1.0, 2.0]]))
    assert np.array_equal(lo, [[0.0, 0.0, 0.0]])
    assert np.array_equal(hi, [[1.0, 1.0, 0.0]])


def test_index_arrays_reject_values_that_do_not_fit_int32():
    """The kernels address memory with these values as int32.

    `np.astype` wraps out-of-range integers silently, so an index that arrives
    as an int64 above `2**31 - 1` becomes a negative offset and reads outside
    the vertex array. `Scene` catches that by range, but only because `i32` let
    the value through first, so `i32` is the one that has to refuse it.
    """
    from mojortreray import _lib

    for value, dtype in [
        (2**31, np.int64),
        (2**31 + 1, np.int64),
        (2**32 + 7, np.uint64),
        (-(2**31) - 1, np.int64),
    ]:
        with pytest.raises(OverflowError):
            _lib.i32(np.array([value], dtype=dtype))
    # The representable range still converts, including the negative end, which
    # is how a deliberately invalid index is expressed.
    assert _lib.i32(np.array([2**31 - 1], np.int64)).tolist() == [2**31 - 1]
    assert _lib.i32(np.array([-1], np.int64)).tolist() == [-1]
    assert _lib.i32(np.array([0], np.uint64)).tolist() == [0]


def test_triangle_bounds_rejects_a_face_index_outside_the_vertex_array():
    """`triangle_bounds` reads `vertices` unchecked.

    A face index one element past the end reads heap memory and returns
    garbage; a large one segfaults the process. The wrapper has to stop it.
    """
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    for faces in (
        np.array([[0, 1, 3]]),
        np.array([[0, 1, 3]], dtype=np.int64),
        np.array([[0, 1, 99999999]], dtype=np.int64),
        np.array([[0, 1, -1]]),
    ):
        with pytest.raises(ValueError):
            mrr.triangle_bounds(vertices, faces)
        with pytest.raises(ValueError):
            mrr.Scene(vertices, faces)


def test_an_empty_scene_is_legal():
    """`min`/`max` over an empty axis has no identity, so the scene bounds are
    defined explicitly rather than reduced from an empty array."""
    scene = mrr.Scene(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    assert scene.ntriangles == 0
    lo, hi = scene.bounds
    assert np.array_equal(lo, np.zeros(3))
    assert np.array_equal(hi, np.zeros(3))
    # And it answers queries, rather than failing at construction.
    rays = np.array([[0.0, 0, 1, 0, 0, -1], [0.0, 0, 1, 0, 1, 0]])
    assert mrr.intersect_rays(rays, scene) == [None, None]
    assert not mrr.occluded_rays(rays, scene).any()
    assert not len(mrr.intersect_all(rays, scene))
    # The BVH over no primitives is a single node marked as holding neither
    # primitives nor children. Traversal has to treat that as an empty tree and
    # return; an all-zero row reads as an interior node whose two children are
    # itself, and the descent never terminates.
    bvh = scene.bvh()
    assert bvh.nint[0, 3] == -1
    t, prim, _, ocount = mrr.traverse_bvh(rays, bvh, scene.soup)
    assert np.all(t == -1.0) and np.all(prim < 0)
    assert not mrr.occluded_bvh(rays, bvh, scene.soup).any()
    again = mrr.BVH(np.zeros((1, 6)), np.zeros((1, 4), np.int32), np.zeros(0, np.int32))
    t, prim, _, _ = mrr.traverse_bvh(rays, again, scene.soup)
    assert np.all(t == -1.0) and np.all(prim < 0)
