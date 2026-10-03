"""Batched ray/scene queries over an explicit candidate list.

The R-tree stage produces, for every ray, a set of candidate primitive ids.
Flattened into one `(ray, primitive)` pair array with a per-ray segment offset,
every remaining stage of the pipeline is a straight-line loop over pairs, and
that is the shape the whole file is written in.

`hit` is carried as a float64 0.0/1.0 mask rather than folded into `t` so the
reductions can read one array for "is there anything here" without a second
pass, and so `MISS` (-1.0) never has to be compared against.
"""

from max.algorithm import parallelize

from mojortreray.common import FPtr, FAR, IPtr, MISS, Vec3
from mojortreray.primitive import SlabRay, ray_triangle_intersect



def ray_triangle_segment(
    rays: FPtr,
    tri: IPtr,
    vertices: FPtr,
    ri: Int,
    start: Int,
    stop: Int,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Bool,
    eps: Float64,
    tout: FPtr,
    hout: FPtr,
):
    """One ray against the candidates the R-tree handed it."""
    var ro = rays.unsafe_offset(6 * ri)
    var origin = Vec3.load(ro)
    var direction = Vec3.load(ro.unsafe_offset(3))
    for k in range(start, stop):
        var v = Int(tri.unsafe_load(k)) * 9
        var t = ray_triangle_intersect(
            Vec3.load(vertices.unsafe_offset(v)),
            Vec3.load(vertices.unsafe_offset(v).unsafe_offset(3)),
            Vec3.load(vertices.unsafe_offset(v).unsafe_offset(6)),
            origin,
            direction,
            tmin,
            tmax,
            cull_backface,
            eps,
        )
        tout.unsafe_store(k, t)
        hout.unsafe_store(k, 1.0 if t >= 0.0 else 0.0)


def ray_triangle_intersect_batch(
    rays: FPtr,
    tri: IPtr,
    offsets: IPtr,
    vertices: FPtr,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Bool,
    eps: Float64,
    tout: FPtr,
    hout: FPtr,
    nray: Int,
    workers: Int,
):
    """Every `(ray, candidate)` pair, written out in pair order.

    Rays are independent and each owns a contiguous run of pairs, so the
    parallel split is over rays and never reorders the output.
    """
    if nray * 32 >= 16384 and workers > 1:
        def work(ri: Int) {imm}:
            ray_triangle_segment(
                rays, tri, vertices, ri,
                Int(offsets.unsafe_load(ri)), Int(offsets.unsafe_load(ri + 1)),
                tmin, tmax, cull_backface, eps, tout, hout,
            )

        parallelize(work, nray, workers)
    else:
        for ri in range(nray):
            ray_triangle_segment(
                rays, tri, vertices, ri,
                Int(offsets.unsafe_load(ri)), Int(offsets.unsafe_load(ri + 1)),
                tmin, tmax, cull_backface, eps, tout, hout,
            )


def ray_aabb_segment(
    rays: FPtr,
    boxes: FPtr,
    box_idx: IPtr,
    ri: Int,
    start: Int,
    stop: Int,
    tmin: Float64,
    tmax: Float64,
    eps: Float64,
    tout: FPtr,
    hout: FPtr,
):
    """One ray against the nodes the R-tree kept for it."""
    var ro = rays.unsafe_offset(6 * ri)
    var rc = SlabRay.load(ro, ro.unsafe_offset(3))
    for k in range(start, stop):
        var b = Int(box_idx.unsafe_load(k)) * 6
        var t = rc.intersect(boxes.unsafe_offset(b), boxes.unsafe_offset(b).unsafe_offset(3), tmin, tmax, eps)
        tout.unsafe_store(k, t)
        hout.unsafe_store(k, 1.0 if t >= 0.0 else 0.0)


def ray_aabb_intersect_batch(
    rays: FPtr,
    boxes: FPtr,
    box_idx: IPtr,
    offsets: IPtr,
    tmin: Float64,
    tmax: Float64,
    eps: Float64,
    tout: FPtr,
    hout: FPtr,
    nray: Int,
    workers: Int,
):
    """The node-culling counterpart of the triangle batch: entry parameter and
    hit flag for every `(ray, node)` pair."""
    if nray * 32 >= 16384 and workers > 1:
        def work(ri: Int) {imm}:
            ray_aabb_segment(
                rays, boxes, box_idx, ri,
                Int(offsets.unsafe_load(ri)), Int(offsets.unsafe_load(ri + 1)),
                tmin, tmax, eps, tout, hout,
            )

        parallelize(work, nray, workers)
    else:
        for ri in range(nray):
            ray_aabb_segment(
                rays, boxes, box_idx, ri,
                Int(offsets.unsafe_load(ri)), Int(offsets.unsafe_load(ri + 1)),
                tmin, tmax, eps, tout, hout,
            )


def segment_closest_hit(
    offsets: IPtr, t: FPtr, hit: FPtr, tout: FPtr, pout: FPtr, nray: Int
):
    """Reduce each ray's segment to the surviving pair with the smallest `t`.
    `pout` holds the *pair* index, not the primitive id: the flat pair list is
    the only place the primitive id still exists.

    Serial on purpose. This is one streaming pass over `t` and `hit` and
    nothing else, so it is already bandwidth-bound; splitting it over rays
    measured slower than the serial loop at every worker count from 2 to 72,
    because a thread launch costs more than the ~20us of work each thread gets
    at the batch sizes real callers pass.
    """
    for ri in range(nray):
        var best = FAR
        var best_k = -1
        for k in range(Int(offsets.unsafe_load(ri)), Int(offsets.unsafe_load(ri + 1))):
            var tk = t.unsafe_load(k)
            if hit.unsafe_load(k) != 0.0 and tk < best:
                best = tk
                best_k = k
        tout.unsafe_store(ri, MISS if best_k < 0 else best)
        pout.unsafe_store(ri, Float64(best_k))


def segment_any_hit(offsets: IPtr, hit: FPtr, flag: IPtr, nray: Int):
    """Occlusion: the first surviving pair ends the segment. Serial for the
    same reason as `segment_closest_hit`, and it stops at the first hit, so it
    usually reads far less than the whole segment."""
    for ri in range(nray):
        var f = 0
        for k in range(Int(offsets.unsafe_load(ri)), Int(offsets.unsafe_load(ri + 1))):
            if hit.unsafe_load(k) != 0.0:
                f = 1
                break
        flag.unsafe_store(ri, Int32(f))


def segment_hit_count(offsets: IPtr, hit: FPtr, count: IPtr, nray: Int):
    """Counts for the counting sort, with a trailing zero slot at `count[nray]`.
    Serial for the same reason as `segment_closest_hit`."""
    for ri in range(nray):
        var c = 0
        for k in range(Int(offsets.unsafe_load(ri)), Int(offsets.unsafe_load(ri + 1))):
            if hit.unsafe_load(k) != 0.0:
                c += 1
        count.unsafe_store(ri, Int32(c))
    count.unsafe_store(nray, Int32(0))


def segment_prefix_sum(count: IPtr, nray: Int) -> Int:
    """In place: `count` becomes the exclusive prefix sum and the return value
    is the total number of hits, which is also left at `count[nray]`."""
    var running = 0
    for i in range(nray):
        var c = Int(count.unsafe_load(i))
        count.unsafe_store(i, Int32(running))
        running += c
    count.unsafe_store(nray, Int32(running))
    return running


def segment_scatter(
    offsets: IPtr,
    slots: IPtr,
    t: FPtr,
    hit: FPtr,
    ray_out: IPtr,
    t_out: FPtr,
    prim_out: FPtr,
    nray: Int,
):
    """Place each ray's hits contiguously in the flat output. Serial for the
    same reason as `segment_closest_hit`, and the writes are already scattered
    to each ray's own slot range."""
    for ri in range(nray):
        var w = Int(slots.unsafe_load(ri))
        for k in range(Int(offsets.unsafe_load(ri)), Int(offsets.unsafe_load(ri + 1))):
            if hit.unsafe_load(k) != 0.0:
                ray_out.unsafe_store(w, Int32(ri))
                t_out.unsafe_store(w, t.unsafe_load(k))
                prim_out.unsafe_store(w, Float64(k))
                w += 1
