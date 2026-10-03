"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric. The bodies below
are pure plumbing -- rebuild the pointer, forward to the kernel in the module
that owns it. Callers own all memory, including scratch, so nothing here
allocates and nothing here can leak.
"""

from mojortreray.bvh import bvh_any_hit, bvh_traverse
from mojortreray.common import FPtr, IPtr
from mojortreray.primitive import (
    box_surface_area,
    ray_aabb_any_hit,
    ray_aabb_intersect,
    ray_sphere_intersect,
    ray_triangle_barycentric,
    ray_triangle_intersect,
    triangle_bounds,
)
from mojortreray.ray import ray_at, ray_make, ray_offset, ray_transform
from mojortreray.traverse import (
    ray_aabb_intersect_batch,
    ray_triangle_intersect_batch,
    segment_any_hit,
    segment_closest_hit,
    segment_hit_count,
    segment_prefix_sum,
    segment_scatter,
)
from mojortreray.common import Vec3


def p(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def q(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


# ------------------------------------------------------------------- ray
@export("mrr_ray_make")
def mrr_ray_make(origin: Int, direction: Int, dst: Int) abi("C"):
    ray_make(p(origin), p(direction), p(dst))


@export("mrr_ray_at")
def mrr_ray_at(origin: Int, direction: Int, t: Float64, dst: Int) abi("C"):
    ray_at(p(origin), p(direction), t, p(dst))


@export("mrr_ray_offset")
def mrr_ray_offset(origin: Int, direction: Int, t: Float64, dst: Int) abi("C"):
    ray_offset(p(origin), p(direction), t, p(dst))


@export("mrr_ray_transform")
def mrr_ray_transform(ray: Int, m: Int, dst: Int) abi("C"):
    ray_transform(p(ray), p(m), p(dst))


# ------------------------------------------------------------ primitive
@export("mrr_ray_aabb_intersect")
def mrr_ray_aabb_intersect(
    origin: Int, direction: Int, lo: Int, hi: Int, tmin: Float64, tmax: Float64, eps: Float64
) abi("C") -> Float64:
    return ray_aabb_intersect(
        p(origin), p(direction), p(lo), p(hi), tmin, tmax, eps
    )


@export("mrr_ray_aabb_any_hit")
def mrr_ray_aabb_any_hit(
    origin: Int, direction: Int, lo: Int, hi: Int, tmin: Float64, tmax: Float64
) abi("C") -> Int:
    return ray_aabb_any_hit(p(origin), p(direction), p(lo), p(hi), tmin, tmax)


@export("mrr_ray_sphere_intersect")
def mrr_ray_sphere_intersect(
    origin: Int,
    direction: Int,
    center: Int,
    radius: Float64,
    tmin: Float64,
    tmax: Float64,
) abi("C") -> Float64:
    return ray_sphere_intersect(
        Vec3.load(p(origin)), Vec3.load(p(direction)), Vec3.load(p(center)),
        radius, tmin, tmax,
    )


@export("mrr_ray_triangle_intersect")
def mrr_ray_triangle_intersect(
    v0: Int,
    v1: Int,
    v2: Int,
    origin: Int,
    direction: Int,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Int,
    eps: Float64,
) abi("C") -> Float64:
    return ray_triangle_intersect(
        Vec3.load(p(v0)), Vec3.load(p(v1)), Vec3.load(p(v2)),
        Vec3.load(p(origin)), Vec3.load(p(direction)),
        tmin, tmax, cull_backface != 0, eps,
    )


@export("mrr_ray_triangle_barycentric")
def mrr_ray_triangle_barycentric(
    v0: Int,
    v1: Int,
    v2: Int,
    origin: Int,
    direction: Int,
    eps: Float64,
    bu: Int,
    bv: Int,
) abi("C"):
    ray_triangle_barycentric(
        Vec3.load(p(v0)), Vec3.load(p(v1)), Vec3.load(p(v2)),
        Vec3.load(p(origin)), Vec3.load(p(direction)),
        eps, p(bu), p(bv),
    )


@export("mrr_triangle_bounds")
def mrr_triangle_bounds(
    vertices: Int, tri: Int, lo: Int, hi: Int, ntri: Int
) abi("C"):
    triangle_bounds(p(vertices), q(tri), p(lo), p(hi), ntri)


@export("mrr_box_surface_area")
def mrr_box_surface_area(lo: Int, hi: Int, area: Int, n: Int) abi("C"):
    box_surface_area(p(lo), p(hi), p(area), n)


# ------------------------------------------------------------- traverse
@export("mrr_ray_triangle_intersect_batch")
def mrr_ray_triangle_intersect_batch(
    rays: Int,
    tri: Int,
    offsets: Int,
    vertices: Int,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Int,
    eps: Float64,
    tout: Int,
    hout: Int,
    nray: Int,
    workers: Int,
) abi("C"):
    ray_triangle_intersect_batch(
        p(rays), q(tri), q(offsets), p(vertices), tmin, tmax,
        cull_backface != 0, eps, p(tout), p(hout), nray, workers,
    )


@export("mrr_ray_aabb_intersect_batch")
def mrr_ray_aabb_intersect_batch(
    rays: Int,
    boxes: Int,
    box_idx: Int,
    offsets: Int,
    tmin: Float64,
    tmax: Float64,
    eps: Float64,
    tout: Int,
    hout: Int,
    nray: Int,
    workers: Int,
) abi("C"):
    ray_aabb_intersect_batch(
        p(rays), p(boxes), q(box_idx), q(offsets), tmin, tmax, eps,
        p(tout), p(hout), nray, workers,
    )


@export("mrr_segment_closest_hit")
def mrr_segment_closest_hit(
    offsets: Int, t: Int, hit: Int, tout: Int, pout: Int, nray: Int
) abi("C"):
    segment_closest_hit(q(offsets), p(t), p(hit), p(tout), p(pout), nray)


@export("mrr_segment_any_hit")
def mrr_segment_any_hit(offsets: Int, hit: Int, flag: Int, nray: Int) abi("C"):
    segment_any_hit(q(offsets), p(hit), q(flag), nray)


@export("mrr_segment_hit_count")
def mrr_segment_hit_count(
    offsets: Int, hit: Int, count: Int, nray: Int
) abi("C"):
    segment_hit_count(q(offsets), p(hit), q(count), nray)


@export("mrr_segment_prefix_sum")
def mrr_segment_prefix_sum(count: Int, nray: Int) abi("C") -> Int:
    return segment_prefix_sum(q(count), nray)


@export("mrr_segment_scatter")
def mrr_segment_scatter(
    offsets: Int, slots: Int, t: Int, hit: Int, ray_out: Int, t_out: Int, prim_out: Int, nray: Int
) abi("C"):
    segment_scatter(
        q(offsets), q(slots), p(t), p(hit), q(ray_out), p(t_out), p(prim_out), nray
    )


# ------------------------------------------------------------------ bvh
@export("mrr_bvh_traverse")
def mrr_bvh_traverse(
    rays: Int,
    nbox: Int,
    nint: Int,
    tris: Int,
    vertices: Int,
    stack: Int,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Int,
    eps: Float64,
    tout: Int,
    pout: Int,
    order: Int,
    ocount: Int,
    nray: Int,
    workers: Int,
    stack_stride: Int,
    order_stride: Int,
) abi("C"):
    bvh_traverse(
        p(rays), p(nbox), q(nint), q(tris), p(vertices), q(stack), tmin, tmax,
        cull_backface != 0, eps, p(tout), p(pout), q(order), q(ocount),
        nray, workers, stack_stride, order_stride,
    )


@export("mrr_bvh_any_hit")
def mrr_bvh_any_hit(
    rays: Int,
    nbox: Int,
    nint: Int,
    tris: Int,
    vertices: Int,
    stack: Int,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Int,
    eps: Float64,
    flag: Int,
    pout: Int,
    nray: Int,
    workers: Int,
    stack_stride: Int,
) abi("C"):
    bvh_any_hit(
        p(rays), p(nbox), q(nint), q(tris), p(vertices), q(stack), tmin, tmax,
        cull_backface != 0, eps, q(flag), p(pout), nray, workers, stack_stride,
    )
