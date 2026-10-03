"""Ray algebra.

A ray is stored flat as six doubles: origin.xyz then direction.xyz. `tmin` and
`tmax` are not part of the record -- they are the interval the caller asks a
query to respect, so a batch can carry one pair per ray without widening the
record to eight doubles.
"""

from std.math import sqrt

from mojortreray.common import FPtr


def ray_make(origin: FPtr, direction: FPtr, dst: FPtr):
    """`Ray.normalize`: the origin is copied, the direction is scaled to unit
    length. A ray is only unit-parameterised if the caller asks for it."""
    for a in range(3):
        dst.unsafe_store(a, origin.unsafe_load(a))
    var len2 = 0.0
    for a in range(3):
        len2 += direction.unsafe_load(a) * direction.unsafe_load(a)
    var inv = 1.0 / sqrt(len2)
    for a in range(3):
        dst.unsafe_store(3 + a, direction.unsafe_load(a) * inv)


def ray_at(origin: FPtr, direction: FPtr, t: Float64, dst: FPtr):
    """`Ray.at`: the point at arc length `t` along the ray."""
    for a in range(3):
        dst.unsafe_store(a, origin.unsafe_load(a) + t * direction.unsafe_load(a))


def ray_offset(origin: FPtr, direction: FPtr, t: Float64, dst: FPtr):
    """`Ray.offset`: the same point, promoted to a fresh origin."""
    for a in range(3):
        dst.unsafe_store(a, origin.unsafe_load(a) + t * direction.unsafe_load(a))


def ray_transform(ray: FPtr, m: FPtr, dst: FPtr):
    """`Ray.transform` by a row-major 4x4.

    The origin picks up the translation column; the direction does not, because
    a ray is a point plus a free vector. That is the whole reason an object-space
    ray cannot simply be transformed like a point.
    """
    for a in range(3):
        var o = m.unsafe_load(4 * a + 3)
        var d = 0.0
        for b in range(3):
            var e = m.unsafe_load(4 * a + b)
            o += e * ray.unsafe_load(b)
            d += e * ray.unsafe_load(3 + b)
        dst.unsafe_store(a, o)
        dst.unsafe_store(3 + a, d)
