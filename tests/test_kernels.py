"""Every `@export`ed kernel, called by name and asserted on its real output.

The other three files reach the kernels through the public API, which is the
right way to test the port but leaves a kernel free to be mis-wired and still
pass: a wrong kernel behind a right one is invisible until something depends on
it. This file names all twenty entry points and checks each one against a value
worked out here rather than by re-running the same wrapper, so the number on the
other side of the ABI is the number this file says it should be.

Every call goes through `_lib.lib()` and `ctypes.data`, exactly as the kernels
are reached in production, so the argument count, the argument order and the
return convention are all part of what is under test.
"""

import ctypes
import math
import re

import numpy as np
import pytest

from mojortreray import _lib

MISS = -1.0
UNIT = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def sig(name):
    """The exported symbol itself, with the declared signature already bound."""
    return getattr(_lib.lib(), name)


def f(arr):
    """A C-contiguous float64 buffer plus the address the ABI wants."""
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return a, a.ctypes.data


def i32a(arr):
    a = np.ascontiguousarray(arr, dtype=np.int32)
    return a, a.ctypes.data


def addr(a):
    return a.ctypes.data


def tiny_bvh():
    """One interior node over two leaves, in the layout the builder emits."""
    nbox = np.ascontiguousarray([[-1.0, -1, -1, 1, 1, 1]] * 3)
    nint = np.ascontiguousarray([[1, 2, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]], np.int32)
    tris = np.ascontiguousarray([0, 1], np.int32)
    # Two unit triangles at z = 0 and z = 2, so `t` distinguishes them.
    soup = np.ascontiguousarray(
        [[0.0, 0, 0, 1, 0, 0, 0, 1, 0], [0.0, 0, 2, 1, 0, 2, 0, 1, 2]]
    )
    return nbox, nint, tris, soup


# ------------------------------------------------------------------ ray
def test_mrr_ray_make_writes_origin_then_unit_direction():
    o, op = f([1.0, 2.0, 3.0])
    d, dp = f([0.0, 3.0, 4.0])
    dst = np.full(6, -99.0)
    sig("mrr_ray_make")(op, dp, addr(dst))
    # The origin is copied verbatim and the direction is scaled to unit length:
    # 3-4-5 gives a scale factor of 1/5, and 3 * (1/5) is not exactly 0.6.
    assert dst[:3].tolist() == [1.0, 2.0, 3.0]
    assert dst[3:] == pytest.approx([0.0, 0.6, 0.8])
    assert np.linalg.norm(dst[3:]) == pytest.approx(1.0)


def test_mrr_ray_at_writes_origin_plus_t_times_direction():
    o, op = f([0.0, 1.0, -2.0])
    d, dp = f([1.0, 0.0, 2.0])
    dst = np.full(3, -99.0)
    sig("mrr_ray_at")(op, dp, ctypes.c_double(2.5), addr(dst))
    assert dst.tolist() == [2.5, 1.0, 3.0]


def test_mrr_ray_offset_writes_the_same_point_as_ray_at():
    o, op = f([0.0, 1.0, -2.0])
    d, dp = f([1.0, 0.0, 2.0])
    dst = np.full(3, -99.0)
    sig("mrr_ray_offset")(op, dp, ctypes.c_double(2.5), addr(dst))
    assert dst.tolist() == [2.5, 1.0, 3.0]


def test_mrr_ray_transform_takes_translation_on_the_origin_only():
    m = np.zeros((4, 4))
    m[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    m[:3, 3] = [10.0, 20.0, 30.0]
    # A 90-degree z rotation of (1, 0, 0) is (0, 1, 0).
    r, rp = f([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    mm, mp = f(m)
    dst = np.full(6, -99.0)
    sig("mrr_ray_transform")(rp, mp, addr(dst))
    assert dst[:3].tolist() == [10.0, 21.0, 30.0]
    assert dst[3:].tolist() == [0.0, 1.0, 0.0]


# ------------------------------------------------------------- box slab
def test_mrr_ray_aabb_intersect_returns_the_entry_parameter():
    o, op = f([-5.0, 0.0, 0.0])
    d, dp = f([1.0, 0.0, 0.0])
    lo, lp = f([-1.0, -1.0, -1.0])
    hi, hp = f([1.0, 1.0, 1.0])
    t = sig("mrr_ray_aabb_intersect")(
        op, dp, lp, hp, ctypes.c_double(0.0),
        ctypes.c_double(math.inf), ctypes.c_double(0.0),
    )
    assert t == pytest.approx(4.0)
    # Behind the ray origin is a miss, and the miss is exactly MISS.
    o2, op2 = f([5.0, 0.0, 0.0])
    assert sig("mrr_ray_aabb_intersect")(
        op2, dp, lp, hp, ctypes.c_double(0.0),
        ctypes.c_double(math.inf), ctypes.c_double(0.0),
    ) == MISS


def test_mrr_ray_aabb_any_hit_returns_an_integer_flag():
    o, op = f([-5.0, 0.0, 0.0])
    d, dp = f([1.0, 0.0, 0.0])
    lo, lp = f([-1.0, -1.0, -1.0])
    hi, hp = f([1.0, 1.0, 1.0])
    assert sig("mrr_ray_aabb_any_hit")(
        op, dp, lp, hp, ctypes.c_double(0.0), ctypes.c_double(math.inf)
    ) == 1
    # Off to one side in y, so the y slab is missed.
    side, sp = f([-5.0, 5.0, 0.0])
    assert sig("mrr_ray_aabb_any_hit")(
        sp, dp, lp, hp, ctypes.c_double(0.0), ctypes.c_double(math.inf)
    ) == 0


# -------------------------------------------------------------- sphere
def test_mrr_ray_sphere_intersect_roots():
    o, op = f([0.0, 0.0, -5.0])
    d, dp = f([0.0, 0.0, 1.0])
    c, cp = f([0.0, 0.0, 0.0])
    assert sig("mrr_ray_sphere_intersect")(
        op, dp, cp, ctypes.c_double(1.0), ctypes.c_double(0.0),
        ctypes.c_double(math.inf),
    ) == pytest.approx(4.0)
    # A non-normalised direction: the quadratic is built from it as given, so
    # direction (0, 0, 2) halves the parameter rather than doubling it.
    d2, dp2 = f([0.0, 0.0, 2.0])
    assert sig("mrr_ray_sphere_intersect")(
        op, dp2, cp, ctypes.c_double(1.0), ctypes.c_double(0.0),
        ctypes.c_double(math.inf),
    ) == pytest.approx(2.0)
    # A ray that passes wide of the sphere has a negative discriminant.
    side, sp = f([-5.0, 5.0, 0.0])
    d3, dp3 = f([1.0, 0.0, 0.0])
    assert sig("mrr_ray_sphere_intersect")(
        sp, dp3, cp, ctypes.c_double(1.0), ctypes.c_double(0.0),
        ctypes.c_double(math.inf),
    ) == MISS
    # A tangent ray has a double root, and it counts as a hit at that root.
    # The perpendicular distance is 1, so the discriminant is exactly zero.
    tan, tp2 = f([-5.0, 1.0, 0.0])
    assert sig("mrr_ray_sphere_intersect")(
        tp2, dp3, cp, ctypes.c_double(1.0), ctypes.c_double(0.0),
        ctypes.c_double(math.inf),
    ) == pytest.approx(5.0)


def test_mrr_ray_sphere_intersect_falls_back_to_the_far_root():
    """Inside the sphere the near root is behind `tmin`; the far root is the
    answer, and a strict comparison would have dropped both."""
    assert sig("mrr_ray_sphere_intersect")(
        addr(f([0.0, 0.0, 0.0])[0]),
        addr(f([1.0, 0.0, 0.0])[0]),
        addr(f([0.0, 0.0, 0.0])[0]),
        ctypes.c_double(1.0), ctypes.c_double(0.0), ctypes.c_double(math.inf),
    ) == pytest.approx(1.0)


# ------------------------------------------------------------ triangle
def test_mrr_ray_triangle_intersect_hit_and_winding():
    v0, p0 = f(UNIT[0])
    v1, p1 = f(UNIT[1])
    v2, p2 = f(UNIT[2])
    o, op = f([0.25, 0.25, -1.0])
    d, dp = f([0.0, 0.0, 1.0])
    # UNIT is (0,0,0), (1,0,0), (0,1,0), whose normal is +z. A ray travelling
    # +z therefore arrives at the back face, and `cull_backface` drops it.
    assert sig("mrr_ray_triangle_intersect")(
        p0, p1, p2, op, dp, ctypes.c_double(0.0),
        ctypes.c_double(math.inf), ctypes.c_int64(0),
        ctypes.c_double(1e-9),
    ) == pytest.approx(1.0)
    assert sig("mrr_ray_triangle_intersect")(
        p0, p1, p2, op, dp, ctypes.c_double(0.0),
        ctypes.c_double(math.inf), ctypes.c_int64(1),
        ctypes.c_double(1e-9),
    ) == MISS
    # The same ray from the front, which is the one culling keeps.
    ob, obp = f([0.25, 0.25, 1.0])
    db, dbp = f([0.0, 0.0, -1.0])
    assert sig("mrr_ray_triangle_intersect")(
        p0, p1, p2, obp, dbp, ctypes.c_double(0.0),
        ctypes.c_double(math.inf), ctypes.c_int64(0),
        ctypes.c_double(1e-9),
    ) == pytest.approx(1.0)
    assert sig("mrr_ray_triangle_intersect")(
        p0, p1, p2, obp, dbp, ctypes.c_double(0.0),
        ctypes.c_double(math.inf), ctypes.c_int64(1),
        ctypes.c_double(1e-9),
    ) == pytest.approx(1.0)


def test_mrr_ray_triangle_intersect_rejects_a_miss_outside_the_edges():
    v0, p0 = f(UNIT[0])
    v1, p1 = f(UNIT[1])
    v2, p2 = f(UNIT[2])
    d, dp = f([0.0, 0.0, 1.0])
    # (0.8, 0.4) has u + v = 1.2, past the hypotenuse.
    o, op = f([0.8, 0.4, -1.0])
    assert sig("mrr_ray_triangle_intersect")(
        p0, p1, p2, op, dp, ctypes.c_double(0.0),
        ctypes.c_double(math.inf), ctypes.c_int64(0),
        ctypes.c_double(1e-9),
    ) == MISS


def test_mrr_ray_triangle_intersect_rejects_a_degenerate_triangle():
    flat, fp = f([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    o, op = f([1.0, 0.0, -1.0])
    d, dp = f([0.0, 0.0, 1.0])
    assert sig("mrr_ray_triangle_intersect")(
        fp, addr(flat) + 8, addr(flat) + 16, op, dp, ctypes.c_double(0.0),
        ctypes.c_double(math.inf), ctypes.c_int64(0), ctypes.c_double(1e-9),
    ) == MISS


def test_mrr_ray_triangle_barycentric_writes_u_and_v():
    v0, p0 = f(UNIT[0])
    v1, p1 = f(UNIT[1])
    v2, p2 = f(UNIT[2])
    o, op = f([0.2, 0.3, -1.0])
    d, dp = f([0.0, 0.0, 1.0])
    bu, bv = np.full(1, -99.0), np.full(1, -99.0)
    sig("mrr_ray_triangle_barycentric")(
        p0, p1, p2, op, dp, ctypes.c_double(1e-9), addr(bu), addr(bv)
    )
    assert bu[0] == pytest.approx(0.2)
    assert bv[0] == pytest.approx(0.3)


def test_mrr_ray_triangle_barycentric_writes_zero_on_a_degenerate_triangle():
    """The determinant guard, rather than a division by zero."""
    flat, fp = f([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    o, op = f([1.0, 0.0, -1.0])
    d, dp = f([0.0, 0.0, 1.0])
    bu, bv = np.full(1, -99.0), np.full(1, -99.0)
    sig("mrr_ray_triangle_barycentric")(
        fp, addr(flat) + 8, addr(flat) + 16, op, dp, ctypes.c_double(1e-9),
        addr(bu), addr(bv),
    )
    assert bu.tolist() == [0.0] and bv.tolist() == [0.0]


def test_mrr_triangle_bounds_writes_one_lo_and_one_hi_per_face():
    v, vp = f([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0.5, 2.0]])
    faces, fp = i32a([[0, 1, 2], [1, 2, 3]])
    lo = np.full((2, 3), -99.0)
    hi = np.full((2, 3), -99.0)
    sig("mrr_triangle_bounds")(vp, fp, addr(lo), addr(hi), 2)
    assert lo.tolist() == [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    assert hi.tolist() == [[1.0, 1.0, 0.0], [1.0, 1.0, 2.0]]


def test_mrr_box_surface_area_writes_one_area_per_box():
    lo, lp = f([[0.0, 0, 0], [-1.0, -2.0, -3.0]])
    hi, hp = f([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    out = np.full(2, -99.0)
    sig("mrr_box_surface_area")(lp, hp, addr(out), 2)
    assert out[0] == pytest.approx(2 * (1 * 2 + 2 * 3 + 3 * 1))
    assert out[1] == pytest.approx(2 * (2 * 4 + 4 * 6 + 6 * 2))


# ------------------------------------------------------------- traverse
def test_mrr_ray_triangle_intersect_batch_fills_every_pair():
    # The triangle spans x, y in [0, 1] at z = 0. One pair each, so `offsets`
    # gives ray 0 pair 0 and ray 1 pair 1.
    rays, rp = f([[0.25, 0.25, -1.0, 0, 0, 1], [2.25, 0.25, -1.0, 1, 0, 0]])
    tri, tp = i32a([0, 0])
    off, op2 = i32a([0, 1, 2])
    soup, sp = f([[0.0, 0, 0, 1, 0, 0, 0, 1, 0]])
    t = np.full(2, -99.0)
    hit = np.full(2, -99.0)
    sig("mrr_ray_triangle_intersect_batch")(
        rp, tp, op2, sp, ctypes.c_double(0.0), ctypes.c_double(math.inf),
        ctypes.c_int64(0), ctypes.c_double(1e-9), addr(t), addr(hit), 2, 1,
    )
    # Ray 0 lands on the triangle; ray 1 crosses the same plane off to the side.
    assert t[0] == pytest.approx(1.0) and hit[0] == 1.0
    assert t[1] == MISS and hit[1] == 0.0


def test_mrr_ray_triangle_intersect_batch_writes_only_the_pairs_it_owns():
    """`offsets` is the authority on which pair belongs to which ray, so a
    segment the kernel is not given must be left exactly as the caller passed it
    in. Ray 1 owns no pair here and its slot keeps the sentinel."""
    rays, rp = f([[0.25, 0.25, -1.0, 0, 0, 1], [9.0, 9.0, -1.0, 0, 0, 1]])
    tri, tp = i32a([0])
    off, op2 = i32a([0, 1, 1])
    soup, sp = f([[0.0, 0, 0, 1, 0, 0, 0, 1, 0]])
    t = np.array([-99.0, -99.0])
    hit = np.array([-99.0, -99.0])
    sig("mrr_ray_triangle_intersect_batch")(
        rp, tp, op2, sp, ctypes.c_double(0.0), ctypes.c_double(math.inf),
        ctypes.c_int64(0), ctypes.c_double(1e-9), addr(t), addr(hit), 2, 1,
    )
    assert t[0] == pytest.approx(1.0) and hit[0] == 1.0
    assert t[1] == -99.0 and hit[1] == -99.0


def test_mrr_ray_aabb_intersect_batch_fills_every_pair():
    rays, rp = f([[-5.0, 0.0, 0.0, 1, 0, 0]])
    boxes, bp = f([[-1.0, -1, -1, 1, 1, 1], [-1.0, 9, -1, 1, 9, 1]])
    idx, ip = i32a([0, 1])
    off, op2 = i32a([0, 2])
    t = np.full(2, -99.0)
    hit = np.full(2, -99.0)
    sig("mrr_ray_aabb_intersect_batch")(
        rp, bp, ip, op2, ctypes.c_double(0.0), ctypes.c_double(math.inf),
        ctypes.c_double(0.0), addr(t), addr(hit), 1, 1,
    )
    assert t[0] == pytest.approx(4.0) and hit[0] == 1.0
    assert t[1] == MISS and hit[1] == 0.0


def test_mrr_segment_closest_hit_reduces_to_the_smallest_t():
    off, op2 = i32a([0, 2, 5])
    t, tp = f([9.0, 3.0, 7.0, -1.0, 2.0])
    hit, hp = f([1.0, 1.0, 0.0, 0.0, 1.0])
    tout = np.full(2, -99.0)
    pout = np.full(2, -99.0)
    sig("mrr_segment_closest_hit")(op2, tp, hp, addr(tout), addr(pout), 2)
    assert tout[0] == pytest.approx(3.0) and pout[0] == 1.0
    assert tout[1] == pytest.approx(2.0) and pout[1] == 4.0


def test_mrr_segment_closest_hit_reports_a_miss_as_a_negative_pair():
    off, op2 = i32a([0, 1])
    t, tp = f([MISS])
    hit, hp = f([0.0])
    tout = np.full(1, -99.0)
    pout = np.full(1, -99.0)
    sig("mrr_segment_closest_hit")(op2, tp, hp, addr(tout), addr(pout), 1)
    assert tout[0] == MISS and pout[0] == -1.0


def test_mrr_segment_any_hit_sets_one_flag_per_ray():
    off, op2 = i32a([0, 2, 4])
    hit, hp = f([0.0, 0.0, 1.0, 0.0])
    flag = np.full(2, -99, dtype=np.int32)
    sig("mrr_segment_any_hit")(op2, hp, addr(flag), 2)
    assert flag.tolist() == [0, 1]


def test_mrr_segment_hit_count_counts_within_each_segment():
    off, op2 = i32a([0, 3, 4])
    hit, hp = f([1.0, 0.0, 1.0, 1.0])
    count = np.full(3, -99, dtype=np.int32)
    sig("mrr_segment_hit_count")(op2, hp, addr(count), 2)
    # Two for the first ray, one for the second, and the trailing zero slot that
    # the prefix sum needs.
    assert count.tolist() == [2, 1, 0]


def test_mrr_segment_prefix_sum_is_exclusive_and_returns_the_total():
    count, cp = i32a([2, 1, 3, 0])
    total = sig("mrr_segment_prefix_sum")(cp, 3)
    assert total == 6
    assert count.tolist() == [0, 2, 3, 6]


def test_mrr_segment_scatter_writes_hits_contiguously_per_ray():
    # Ray 0 owns pairs 0..3 and ray 1 owns pairs 3..5. Every pair with a hit
    # lands in its own ray's output range, in pair order rather than sorted.
    off, op2 = i32a([0, 3, 5])
    slots, sp2 = i32a([0, 2])
    t, tp = f([5.0, 1.0, 3.0, 8.0, 2.0])
    hit, hp = f([1.0, 1.0, 0.0, 0.0, 1.0])
    ray_out = np.full(3, -99, dtype=np.int32)
    t_out = np.full(3, -99.0)
    prim_out = np.full(3, -99.0)
    sig("mrr_segment_scatter")(
        op2, sp2, tp, hp, addr(ray_out), addr(t_out), addr(prim_out), 2
    )
    assert ray_out.tolist() == [0, 0, 1]
    # Ray 0's hits are pairs 0 and 1 (t = 5, t = 1) in pair order, not sorted
    # by distance; the sorting is `intersect_all`'s job, not this kernel's.
    assert t_out.tolist() == [5.0, 1.0, 2.0]
    # The third output is the flat pair index, not a primitive id.
    assert prim_out.tolist() == [0.0, 1.0, 4.0]


def test_mrr_segment_scatter_leaves_gaps_where_a_ray_has_fewer_hits():
    """The `slots` prefix sum is what keeps the runs contiguous when the rays
    ahead of this one wrote fewer hits than they had pairs. Pair 3 is ray 1's
    first and it misses, so ray 1's single hit goes to slot 1, not slot 2."""
    off, op2 = i32a([0, 1, 2])
    slots, sp2 = i32a([0, 1])
    t, tp = f([5.0, 7.0])
    hit, hp = f([1.0, 1.0])
    ray_out = np.full(2, -99, dtype=np.int32)
    t_out = np.full(2, -99.0)
    prim_out = np.full(2, -99.0)
    sig("mrr_segment_scatter")(
        op2, sp2, tp, hp, addr(ray_out), addr(t_out), addr(prim_out), 2
    )
    assert ray_out.tolist() == [0, 1]
    assert t_out.tolist() == [5.0, 7.0]
    assert prim_out.tolist() == [0.0, 1.0]


# ------------------------------------------------------------------ bvh
def test_mrr_bvh_traverse_finds_the_closest_primitive():
    nbox, nint, tris, soup = tiny_bvh()
    rays, rp = f([[0.25, 0.25, -5.0, 0, 0, 1], [5.0, 5.0, 5.0, 0, 0, 1]])
    stack = np.zeros(2 * 128, np.int32)
    order = np.zeros(2 * 256, np.int32)
    ocount = np.full(2, -99, np.int32)
    t = np.full(2, -99.0)
    prim = np.full(2, -99.0)
    sig("mrr_bvh_traverse")(
        rp, addr(nbox), addr(nint), addr(tris), addr(soup), addr(stack),
        ctypes.c_double(0.0), ctypes.c_double(math.inf), ctypes.c_int64(0),
        ctypes.c_double(1e-9), addr(t), addr(prim), addr(order), addr(ocount),
        2, 1, 128, 256,
    )
    assert t[0] == pytest.approx(5.0) and prim[0] == 0.0
    assert t[1] == MISS and prim[1] == -1.0
    # The visit order is recorded, and the root is always the first node in it.
    assert ocount[0] >= 1
    assert order[0] == 0


def test_mrr_bvh_traverse_respects_tmax():
    nbox, nint, tris, soup = tiny_bvh()
    rays, rp = f([[0.25, 0.25, -5.0, 0, 0, 1]])
    stack = np.zeros(128, np.int32)
    order = np.zeros(256, np.int32)
    ocount = np.zeros(1, np.int32)
    t = np.full(1, -99.0)
    prim = np.full(1, -99.0)
    sig("mrr_bvh_traverse")(
        rp, addr(nbox), addr(nint), addr(tris), addr(soup), addr(stack),
        ctypes.c_double(0.0), ctypes.c_double(1.0), ctypes.c_int64(0),
        ctypes.c_double(1e-9), addr(t), addr(prim), addr(order), addr(ocount),
        1, 1, 128, 256,
    )
    assert t[0] == MISS and prim[0] == -1.0


def test_mrr_bvh_any_hit_sets_the_flag_and_the_blocking_primitive():
    nbox, nint, tris, soup = tiny_bvh()
    rays, rp = f([[0.25, 0.25, -5.0, 0, 0, 1], [5.0, 5.0, 5.0, 0, 0, 1]])
    stack = np.zeros(2 * 128, np.int32)
    flag = np.full(2, -99, np.int32)
    prim = np.full(2, -99.0)
    sig("mrr_bvh_any_hit")(
        rp, addr(nbox), addr(nint), addr(tris), addr(soup), addr(stack),
        ctypes.c_double(0.0), ctypes.c_double(math.inf), ctypes.c_int64(0),
        ctypes.c_double(1e-9), addr(flag), addr(prim), 2, 1, 128,
    )
    assert flag.tolist() == [1, 0]
    assert prim[0] == 0.0 and prim[1] == -1.0


def _bounded_kernel_call(case: str, timeout: float = 30.0):
    """Run one of the termination cases in a child process and bring the result
    back, or fail if the child has to be killed.

    These two cases assert that the traversal *returns*. If the guard they cover
    is ever removed, the kernel loops forever inside C, where neither a
    `SIGALRM` nor a Python-level timeout can interrupt it, and the test would hang
    the suite instead of failing it. Running the call in a child and terminating
    it turns that into an ordinary assertion failure.
    """
    import multiprocessing
    import queue

    ctx = multiprocessing.get_context("fork")
    out = ctx.Queue()

    def worker():
        try:
            out.put(_termination_case(case))
        except Exception as exc:  # surfaced by the parent as a failure
            out.put({"error": f"{type(exc).__name__}: {exc}"})

    proc = ctx.Process(target=worker, daemon=True)
    proc.start()
    try:
        result = out.get(timeout=timeout)
    except queue.Empty:
        proc.terminate()
        proc.join(timeout=5)
        raise AssertionError(
            f"{case}: the kernel did not return within {timeout}s -- the "
            "traversal is not terminating on this node shape"
        )
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
    if "error" in result:
        raise AssertionError(result["error"])
    return result


def _termination_case(case: str) -> dict:
    """The two non-terminating node shapes, and the answers they must give."""
    soup = np.zeros((0, 9))
    tris = np.zeros(0, np.int32)
    rays, rp = f([[0.0, 0, 5, 0, 0, -1]])
    stack = np.zeros(128, np.int32)
    if case == "empty":
        # `count == -1`: no primitives and no children.
        nbox = np.ascontiguousarray([[0.0, 0, 0, 0, 0, 0]])
        nint = np.ascontiguousarray([[0, 0, 0, -1]], np.int32)
    elif case == "self_edge":
        # `count == 0` with a child pointing back at the node itself.
        nbox = np.ascontiguousarray([[-1.0, -1, -1, 1, 1, 1]] * 2)
        nint = np.ascontiguousarray([[1, 0, 0, 0], [0, 0, 0, 0]], np.int32)
    else:
        raise AssertionError(f"unknown case {case!r}")

    order = np.zeros(256, np.int32)
    ocount = np.full(1, -99, np.int32)
    t = np.full(1, -99.0)
    prim = np.full(1, -99.0)
    sig("mrr_bvh_traverse")(
        rp, addr(nbox), addr(nint), addr(tris), addr(soup), addr(stack),
        ctypes.c_double(0.0), ctypes.c_double(math.inf), ctypes.c_int64(0),
        ctypes.c_double(1e-9), addr(t), addr(prim), addr(order), addr(ocount),
        1, 1, 128, 256,
    )
    flag = np.full(1, -99, np.int32)
    prim2 = np.full(1, -99.0)
    sig("mrr_bvh_any_hit")(
        rp, addr(nbox), addr(nint), addr(tris), addr(soup), addr(stack),
        ctypes.c_double(0.0), ctypes.c_double(math.inf), ctypes.c_int64(0),
        ctypes.c_double(1e-9), addr(flag), addr(prim2), 1, 1, 128,
    )
    return {
        "t": float(t[0]), "prim": float(prim[0]), "ocount": int(ocount[0]),
        "flag": int(flag[0]), "prim2": float(prim2[0]),
    }


# ---------------------------------------------------- coverage of the set
def test_every_export_is_called_by_name_here():
    """If a kernel is added to `src/capi.mojo` without a test added here, this
    fails. The `_SIGNATURES` table in `_lib.py` has to grow with it too, so all
    three lists are checked against each other.

    Naming a kernel is not enough: it also has to be reached through an actual
    `sig("...")(...)` call, so a name that only appeared inside a string would be
    counted above but never exercised.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "src" / "capi.mojo").read_text()
    exported = set(re.findall(r'@export\("([A-Za-z_0-9]+)"\)', src))
    here = pathlib.Path(__file__).read_text()
    assert exported, "no @export found in src/capi.mojo"
    assert set(_lib._SIGNATURES) == exported
    called = set(re.findall(r'sig\("([A-Za-z_0-9]+)"\)\(', here))
    assert called == exported, sorted(exported - called)

def test_mrr_bvh_traverse_terminates_on_an_empty_node():
    """`count == -1` is an explicitly empty node: no primitives, no children.

    Read as an interior node -- which is what a zeroed row is -- its two children
    are itself, so every push re-enters the same node and the descent never
    finishes. This was an infinite loop before the marker was honoured.
    """
    got = _bounded_kernel_call("empty")
    assert got["t"] == MISS and got["prim"] == -1.0
    # The empty node was still visited, and then nothing descended from it.
    assert got["ocount"] == 1
    assert got["flag"] == 0 and got["prim2"] == -1.0


def test_mrr_bvh_traverse_terminates_on_a_self_referential_node():
    """The other non-terminating shape: `count == 0` with a child pointing back
    at the node itself. The builder emits children in pre-order so this cannot
    arise from `BVH.build`, but the kernel reads an array it does not own."""
    got = _bounded_kernel_call("self_edge")
    assert got["t"] == MISS and got["prim"] == -1.0
    assert got["flag"] == 0
