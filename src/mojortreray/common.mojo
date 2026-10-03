"""Shared aliases for the C ABI surface and the small vector type the
intersection routines are written against.

`@export` rejects parametric functions and a pointer with an inferred origin is
parametric, so every exported entry point takes an `Int` address and rebuilds
one of these pointers inside the body. Nothing here allocates.
"""

from std.math import sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]

comptime W = 4
comptime MISS = -1.0
comptime FAR = 1.7976931348623157e308


struct Vec3:
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def load(p: FPtr) -> Vec3:
        return Vec3(p.unsafe_load(0), p.unsafe_load(1), p.unsafe_load(2))

    @staticmethod
    def splat(v: Float64) -> Vec3:
        return Vec3(v, v, v)

    def store(self, p: FPtr):
        p.unsafe_store(0, self.x)
        p.unsafe_store(1, self.y)
        p.unsafe_store(2, self.z)

    def __sub__(self, o: Vec3) -> Vec3:
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)


def dot3(a: Vec3, b: Vec3) -> Float64:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cross3(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def length3(a: Vec3) -> Float64:
    return sqrt(dot3(a, a))
