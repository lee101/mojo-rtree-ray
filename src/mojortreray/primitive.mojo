"""Ray/primitive intersection.

Every routine returns the ray parameter of the *closest* intersection inside
`[tmin, tmax]`, or `MISS` (-1.0) when there is none. `tmin` is the only
convention that matters: an origin inside the geometry hits at `tmin`, not at a
negative parameter.

The box routines take pointers because the callers that matter -- the BVH
interior test and the batched node cull -- already hold a pointer into a node
array, and copying a Vec3 out of it just to read it back is waste. The triangle
routine takes Vec3 because Moller-Trumbore is a fixed, fully unrolled 3-vector
expression with nothing to loop over.
"""

from std.math import sqrt

from mojortreray.common import FPtr, IPtr, MISS, Vec3, cross3, dot3


struct SlabRay:
    """One ray, prepared for repeated slab tests against many boxes.

    The two hot callers -- the batched node cull and the BVH interior test --
    apply the same ray to many boxes, so `1.0 / direction[a]` and the
    "is this axis parallel" test are identical every time. Building them once
    here removes three divisions from every `(ray, box)` pair. Building from the
    pointer is one pass; the fields are read as plain locals in the inner test.
    """
    var ox: Float64
    var oy: Float64
    var oz: Float64
    var ix: Float64
    var iy: Float64
    var iz: Float64
    var par_x: Bool
    var par_y: Bool
    var par_z: Bool

    def __init__(
        out self,
        ox: Float64,
        oy: Float64,
        oz: Float64,
        ix: Float64,
        iy: Float64,
        iz: Float64,
        par_x: Bool,
        par_y: Bool,
        par_z: Bool,
    ):
        self.ox = ox
        self.oy = oy
        self.oz = oz
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.par_x = par_x
        self.par_y = par_y
        self.par_z = par_z

    @staticmethod
    def load(origin: FPtr, direction: FPtr) -> SlabRay:
        var dx = direction.unsafe_load(0)
        var dy = direction.unsafe_load(1)
        var dz = direction.unsafe_load(2)
        var par_x = dx == 0.0
        var par_y = dy == 0.0
        var par_z = dz == 0.0
        var ix = 0.0
        var iy = 0.0
        var iz = 0.0
        if not par_x:
            ix = 1.0 / dx
        if not par_y:
            iy = 1.0 / dy
        if not par_z:
            iz = 1.0 / dz
        return SlabRay(
            origin.unsafe_load(0), origin.unsafe_load(1), origin.unsafe_load(2),
            ix, iy, iz, par_x, par_y, par_z,
        )

    def intersect(self, lo: FPtr, hi: FPtr, tmin: Float64, tmax: Float64, eps: Float64) -> Float64:
        """The slab test proper, over the prepared ray. Three axes, unrolled.

        The zero-direction component is branched on rather than left to the usual
        `1.0 / 0.0` trick: multiplying an infinite reciprocal by a zero offset
        produces a NaN, NaN fails every comparison, and the box is then reported
        as a miss precisely when the ray travels inside that slab. Everything
        downstream of `1.0 / d` is bit-identical to computing it per box.
        """
        var t0 = tmin
        var t1 = tmax

        if self.par_x:
            if self.ox < lo.unsafe_load(0) or self.ox > hi.unsafe_load(0):
                return MISS
        else:
            var ta = (lo.unsafe_load(0) - self.ox) * self.ix
            var tb = (hi.unsafe_load(0) - self.ox) * self.ix
            if ta > tb:
                var swap = ta
                ta = tb
                tb = swap
            t0 = max(t0, ta)
            t1 = min(t1, tb)
            if t0 > t1 + eps:
                return MISS

        if self.par_y:
            if self.oy < lo.unsafe_load(1) or self.oy > hi.unsafe_load(1):
                return MISS
        else:
            var ta = (lo.unsafe_load(1) - self.oy) * self.iy
            var tb = (hi.unsafe_load(1) - self.oy) * self.iy
            if ta > tb:
                var swap = ta
                ta = tb
                tb = swap
            t0 = max(t0, ta)
            t1 = min(t1, tb)
            if t0 > t1 + eps:
                return MISS

        if self.par_z:
            if self.oz < lo.unsafe_load(2) or self.oz > hi.unsafe_load(2):
                return MISS
        else:
            var ta = (lo.unsafe_load(2) - self.oz) * self.iz
            var tb = (hi.unsafe_load(2) - self.oz) * self.iz
            if ta > tb:
                var swap = ta
                ta = tb
                tb = swap
            t0 = max(t0, ta)
            t1 = min(t1, tb)
            if t0 > t1 + eps:
                return MISS

        if t1 < 0.0:
            return MISS
        return t0


def ray_aabb_intersect(
    origin: FPtr, direction: FPtr, lo: FPtr, hi: FPtr, tmin: Float64, tmax: Float64, eps: Float64
) -> Float64:
    """Slab test. Entry parameter of the ray/box overlap, or `MISS`.

    A single-box convenience wrapper over the prepared-ray test, so there is
    exactly one implementation of the slab arithmetic.
    """
    return SlabRay.load(origin, direction).intersect(lo, hi, tmin, tmax, eps)


def ray_aabb_any_hit(
    origin: FPtr, direction: FPtr, lo: FPtr, hi: FPtr, tmin: Float64, tmax: Float64
) -> Int:
    """The same slabs, but the loop only has to prove that one overlap exists,
    so it returns from the first successful axis instead of tracking an entry."""
    var t0 = tmin
    var t1 = tmax
    for a in range(3):
        var d = direction.unsafe_load(a)
        var o = origin.unsafe_load(a)
        if d == 0.0:
            if o < lo.unsafe_load(a) or o > hi.unsafe_load(a):
                return 0
            continue
        var inv = 1.0 / d
        var ta = (lo.unsafe_load(a) - o) * inv
        var tb = (hi.unsafe_load(a) - o) * inv
        if ta > tb:
            var swap = ta
            ta = tb
            tb = swap
        t0 = max(t0, ta)
        t1 = min(t1, tb)
        if t0 > t1:
            return 0
    if t1 < 0.0:
        return 0
    return 1


def ray_sphere_intersect(
    origin: Vec3,
    direction: Vec3,
    center: Vec3,
    radius: Float64,
    tmin: Float64,
    tmax: Float64,
) -> Float64:
    """Quadratic in the ray parameter. The direction need not be normalised;
    the coefficients are built from it as given."""
    var oc = origin - center
    var a = dot3(direction, direction)
    var b = 2.0 * dot3(oc, direction)
    var c = dot3(oc, oc) - radius * radius
    var disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return MISS
    var sq = sqrt(disc)
    var t0 = (-b - sq) / (2.0 * a)
    # `>=`, not `>`: a ray that starts exactly on the surface is already
    # inside the sphere, so the near root at exactly `tmin` is the hit. A
    # strict comparison drops it and reports the far root instead.
    if t0 >= tmin and t0 <= tmax:
        return t0
    var t1 = (-b + sq) / (2.0 * a)
    if t1 >= tmin and t1 <= tmax:
        return t1
    return MISS


def ray_triangle_intersect(
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    origin: Vec3,
    direction: Vec3,
    tmin: Float64,
    tmax: Float64,
    cull_backface: Bool,
    eps: Float64,
) -> Float64:
    """Moller-Trumbore, the `Fast MinimumStorage` form: one determinant, no
    barycentric rescaling, and both edge tests run before the parameter is
    recovered.

    With `cull_backface` the triangle is one-sided and only front faces count,
    which is what a rasteriser wants; otherwise the sign of the determinant only
    has to be non-degenerate and the triangle is hit from both sides.
    """
    var kE1 = v1 - v0
    var kE2 = v2 - v0
    var pvec = cross3(direction, kE2)
    var det = dot3(kE1, pvec)
    if cull_backface:
        if det <= 0.0:
            return MISS
    else:
        if det > -eps and det < eps:
            return MISS
    var inv_det = 1.0 / det
    var tvec = origin - v0
    var u = dot3(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return MISS
    var qvec = cross3(tvec, kE1)
    var v = dot3(direction, qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return MISS
    var t = dot3(kE2, qvec) * inv_det
    if t < tmin or t > tmax:
        return MISS
    return t


def ray_triangle_barycentric(
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    origin: Vec3,
    direction: Vec3,
    eps: Float64,
    bu: FPtr,
    bv: FPtr,
):
    """The `(u, v)` of an already-accepted hit, so a shaded point can be
    reconstructed without redoing the whole intersection."""
    var kE1 = v1 - v0
    var kE2 = v2 - v0
    var pvec = cross3(direction, kE2)
    var det = dot3(kE1, pvec)
    if det > -eps and det < eps:
        bu.unsafe_store(0, 0.0)
        bv.unsafe_store(0, 0.0)
        return
    var inv_det = 1.0 / det
    var tvec = origin - v0
    bu.unsafe_store(0, dot3(tvec, pvec) * inv_det)
    var qvec = cross3(tvec, kE1)
    bv.unsafe_store(0, dot3(direction, qvec) * inv_det)


def triangle_bounds(vertices: FPtr, tri: IPtr, lo: FPtr, hi: FPtr, ntri: Int):
    """Per-triangle AABB of an indexed mesh, the shape the R-tree is built over.

    `vertices` is `(n, 3)` and `tri` is `(ntri, 3)` of vertex indices, so a
    vertex shared by many faces is stored once and read nine times per face
    that uses it. The traversal kernels want the opposite -- a flat soup with
    no index indirection -- and `Scene` hands them one.
    """
    for f in range(ntri):
        var v = Int(tri.unsafe_load(3 * f)) * 3
        for a in range(3):
            var mn = vertices.unsafe_load(v + a)
            var mx = mn
            for w in range(1, 3):
                var s = vertices.unsafe_load(Int(tri.unsafe_load(3 * f + w)) * 3 + a)
                mn = min(mn, s)
                mx = max(mx, s)
            lo.unsafe_store(3 * f + a, mn)
            hi.unsafe_store(3 * f + a, mx)


def box_surface_area(lo: FPtr, hi: FPtr, area: FPtr, n: Int):
    """The SAH cost estimate, on the assumption that a node visit costs what
    visiting does: in proportion to the surface a hit is likely to be on."""
    for f in range(n):
        var dx = hi.unsafe_load(3 * f) - lo.unsafe_load(3 * f)
        var dy = hi.unsafe_load(3 * f + 1) - lo.unsafe_load(3 * f + 1)
        var dz = hi.unsafe_load(3 * f + 2) - lo.unsafe_load(3 * f + 2)
        area.unsafe_store(f, 2.0 * (dx * dy + dy * dz + dz * dx))
