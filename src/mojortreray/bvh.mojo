"""Closest-hit and any-hit traversal of a flat, linearly laid out BVH.

The hierarchy is two arrays. `nbox` is `(nodes, 6)` of `lo.xyz hi.xyz`. `nint`
is `(nodes, 4)` of `left, right, first, count`, where `count > 0` marks a leaf
whose `count` primitive ids start at `first` in `tris`, `count == 0` marks
an interior node with exactly two children, and `count < 0` marks an
explicitly empty node. `tris` is the permutation of
primitive ids the builder produced, and is what makes the layout linear: a
leaf's primitives are adjacent, so the inner loop is a walk over `tris`.

The traversal keeps the classic invariant -- a node is pushed only after the
ray/box slab test has already succeeded against the current best `t`, so the
stack only ever holds nodes that are still worth testing, and the nearer child
is always popped first.

Both descents read `nint` as an array they do not own, so termination cannot
rest on the builder having produced a well-formed tree. A node whose child
index is itself is skipped, which is the only way an interior node can cycle.
"""

from max.algorithm import parallelize

from mojortreray.common import FPtr, IPtr, MISS, Vec3
from mojortreray.primitive import SlabRay, ray_triangle_intersect


def bvh_traverse(
    rays: FPtr,
    nbox: FPtr,
    nint: IPtr,
    tris: IPtr,
    vertices: FPtr,
    stack: IPtr,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Bool,
    eps: Float64,
    tout: FPtr,
    pout: FPtr,
    order: IPtr,
    ocount: IPtr,
    nray: Int,
    workers: Int,
    stack_stride: Int,
    order_stride: Int,
):
    """Closest hit per ray. `pout` is the primitive id, or -1 for a miss.

    `order`/`ocount` record which nodes each ray visited, in visit order, into
    a per-ray strided region; that is how the traversal order is inspected and
    asserted without changing the traversal itself.
    """
    if nray >= 256 and workers > 1:
        def work(ri: Int) {imm}:
            bvh_traverse_one(
                rays, nbox, nint, tris, vertices, stack, ri, tmin, tmax,
                cull_backface, eps, tout, pout, order, ocount,
                stack_stride, order_stride,
            )

        parallelize(work, nray, workers)
    else:
        for ri in range(nray):
            bvh_traverse_one(
                rays, nbox, nint, tris, vertices, stack, ri, tmin, tmax,
                cull_backface, eps, tout, pout, order, ocount,
                stack_stride, order_stride,
            )


def bvh_traverse_one(
    rays: FPtr,
    nbox: FPtr,
    nint: IPtr,
    tris: IPtr,
    vertices: FPtr,
    stack: IPtr,
    ri: Int,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Bool,
    eps: Float64,
    tout: FPtr,
    pout: FPtr,
    order: IPtr,
    ocount: IPtr,
    stack_stride: Int,
    order_stride: Int,
):
    var ro = rays.unsafe_offset(6 * ri)
    var origin = Vec3.load(ro)
    var direction = Vec3.load(ro.unsafe_offset(3))
    var rc = SlabRay.load(ro, ro.unsafe_offset(3))
    var base = ri * stack_stride
    var obase = ri * order_stride
    var t = tmax
    var best = -1
    var sp = 0
    var nvis = 0
    stack.unsafe_store(base, Int32(0))
    sp = 1
    while sp > 0:
        sp -= 1
        var node = Int(stack.unsafe_load(base + sp))
        if nvis < order_stride:
            order.unsafe_store(obase + nvis, Int32(node))
        nvis += 1
        var nb = nbox.unsafe_offset(6 * node)
        var tnear = rc.intersect(nb, nb.unsafe_offset(3), tmin, t, eps)
        if tnear < 0.0:
            continue
        var count = Int(nint.unsafe_load(4 * node + 3))
        if count > 0:
            var first = Int(nint.unsafe_load(4 * node + 2))
            for k in range(count):
                var tri = Int(tris.unsafe_load(first + k))
                var v = tri * 9
                var th = ray_triangle_intersect(
                    Vec3.load(vertices.unsafe_offset(v)),
                    Vec3.load(vertices.unsafe_offset(v).unsafe_offset(3)),
                    Vec3.load(vertices.unsafe_offset(v).unsafe_offset(6)),
                    origin,
                    direction,
                    tmin,
                    t,
                    cull_backface,
                    eps,
                )
                if th >= 0.0 and th < t:
                    t = th
                    best = tri
        elif count < 0:
            # An explicitly empty tree: a node marked as holding no primitives and
            # no children. `BVH.build` emits one for a zero-primitive scene, and
            # without this it reads as an interior node whose two children are
            # itself, which never terminates.
            continue
        else:
            var left = Int(nint.unsafe_load(4 * node))
            var right = Int(nint.unsafe_load(4 * node + 1))
            # A child that is the node itself is a self-edge, and pushing it
            # would re-enter this same node forever. The builder emits children
            # in pre-order, so a child index is always greater than its parent's
            # and this can never reject a real descent.
            if left == node or right == node:
                continue
            var lb = nbox.unsafe_offset(6 * left)
            var rb = nbox.unsafe_offset(6 * right)
            var tl = rc.intersect(lb, lb.unsafe_offset(3), tmin, t, eps)
            var tr = rc.intersect(rb, rb.unsafe_offset(3), tmin, t, eps)
            if tl >= 0.0 and tr >= 0.0:
                if tl <= tr:
                    if sp < stack_stride - 2:
                        stack.unsafe_store(base + sp, Int32(right))
                        sp += 1
                        stack.unsafe_store(base + sp, Int32(left))
                        sp += 1
                else:
                    if sp < stack_stride - 2:
                        stack.unsafe_store(base + sp, Int32(left))
                        sp += 1
                        stack.unsafe_store(base + sp, Int32(right))
                        sp += 1
            elif tl >= 0.0:
                if sp < stack_stride - 1:
                    stack.unsafe_store(base + sp, Int32(left))
                    sp += 1
            elif tr >= 0.0:
                if sp < stack_stride - 1:
                    stack.unsafe_store(base + sp, Int32(right))
                    sp += 1
    tout.unsafe_store(ri, MISS if best < 0 else t)
    pout.unsafe_store(ri, Float64(best))
    ocount.unsafe_store(ri, Int32(min(nvis, order_stride)))


def bvh_any_hit(
    rays: FPtr,
    nbox: FPtr,
    nint: IPtr,
    tris: IPtr,
    vertices: FPtr,
    stack: IPtr,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Bool,
    eps: Float64,
    flag: IPtr,
    pout: FPtr,
    nray: Int,
    workers: Int,
    stack_stride: Int,
):
    """Occlusion. Identical descent, but the first surviving primitive ends the
    ray: `flag` is 1 when something blocks it and `pout` is that primitive."""
    if nray >= 256 and workers > 1:
        def work(ri: Int) {imm}:
            bvh_any_hit_one(
                rays, nbox, nint, tris, vertices, stack, ri, tmin, tmax,
                cull_backface, eps, flag, pout, stack_stride,
            )

        parallelize(work, nray, workers)
    else:
        for ri in range(nray):
            bvh_any_hit_one(
                rays, nbox, nint, tris, vertices, stack, ri, tmin, tmax,
                cull_backface, eps, flag, pout, stack_stride,
            )


def bvh_any_hit_one(
    rays: FPtr,
    nbox: FPtr,
    nint: IPtr,
    tris: IPtr,
    vertices: FPtr,
    stack: IPtr,
    ri: Int,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Bool,
    eps: Float64,
    flag: IPtr,
    pout: FPtr,
    stack_stride: Int,
):
    var ro = rays.unsafe_offset(6 * ri)
    var origin = Vec3.load(ro)
    var direction = Vec3.load(ro.unsafe_offset(3))
    var rc = SlabRay.load(ro, ro.unsafe_offset(3))
    var base = ri * stack_stride
    var t = tmax
    var found = -1
    var sp = 0
    stack.unsafe_store(base, Int32(0))
    sp = 1
    while sp > 0:
        sp -= 1
        var node = Int(stack.unsafe_load(base + sp))
        var nb = nbox.unsafe_offset(6 * node)
        var tnear = rc.intersect(nb, nb.unsafe_offset(3), tmin, t, eps)
        if tnear < 0.0:
            continue
        var count = Int(nint.unsafe_load(4 * node + 3))
        if count > 0:
            var first = Int(nint.unsafe_load(4 * node + 2))
            for k in range(count):
                var tri = Int(tris.unsafe_load(first + k))
                var v = tri * 9
                var th = ray_triangle_intersect(
                    Vec3.load(vertices.unsafe_offset(v)),
                    Vec3.load(vertices.unsafe_offset(v).unsafe_offset(3)),
                    Vec3.load(vertices.unsafe_offset(v).unsafe_offset(6)),
                    origin,
                    direction,
                    tmin,
                    t,
                    cull_backface,
                    eps,
                )
                if th >= 0.0 and th < t:
                    t = th
                    found = tri
            if found >= 0:
                break
        elif count < 0:
            # An explicitly empty tree; see `bvh_traverse_one`.
            continue
        else:
            var left = Int(nint.unsafe_load(4 * node))
            var right = Int(nint.unsafe_load(4 * node + 1))
            # A self-edge would re-enter this node forever.
            if left == node or right == node:
                continue
            var lb = nbox.unsafe_offset(6 * left)
            var rb = nbox.unsafe_offset(6 * right)
            var tl = rc.intersect(lb, lb.unsafe_offset(3), tmin, t, eps)
            var tr = rc.intersect(rb, rb.unsafe_offset(3), tmin, t, eps)
            if tl >= 0.0 and tr >= 0.0:
                if tl <= tr:
                    if sp < stack_stride - 2:
                        stack.unsafe_store(base + sp, Int32(right))
                        sp += 1
                        stack.unsafe_store(base + sp, Int32(left))
                        sp += 1
                else:
                    if sp < stack_stride - 2:
                        stack.unsafe_store(base + sp, Int32(left))
                        sp += 1
                        stack.unsafe_store(base + sp, Int32(right))
                        sp += 1
            elif tl >= 0.0:
                if sp < stack_stride - 1:
                    stack.unsafe_store(base + sp, Int32(left))
                    sp += 1
            elif tr >= 0.0:
                if sp < stack_stride - 1:
                    stack.unsafe_store(base + sp, Int32(right))
                    sp += 1
    flag.unsafe_store(ri, Int32(1 if found >= 0 else 0))
    pout.unsafe_store(ri, Float64(found))
