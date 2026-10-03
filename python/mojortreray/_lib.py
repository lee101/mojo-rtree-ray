"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJORTRE_RAY_LIB") or os.path.join(
    ROOT, "dist", "libmojo-rtree-ray.so"
)

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "mrr_ray_make": ([I, I, I], None),
    "mrr_ray_at": ([I, I, F, I], None),
    "mrr_ray_offset": ([I, I, F, I], None),
    "mrr_ray_transform": ([I, I, I], None),
    "mrr_ray_aabb_intersect": ([I, I, I, I, F, F, F], F),
    "mrr_ray_aabb_any_hit": ([I, I, I, I, F, F], I),
    "mrr_ray_sphere_intersect": ([I, I, I, F, F, F], F),
    "mrr_ray_triangle_intersect": ([I] * 5 + [F, F, I, F], F),
    "mrr_ray_triangle_barycentric": ([I] * 5 + [F, I, I], None),
    "mrr_triangle_bounds": ([I, I, I, I, I], None),
    "mrr_box_surface_area": ([I, I, I, I], None),
    "mrr_ray_triangle_intersect_batch": ([I] * 4 + [F, F, I, F, I, I, I, I], None),
    "mrr_ray_aabb_intersect_batch": ([I] * 4 + [F, F, F] + [I] * 4, None),
    "mrr_segment_closest_hit": ([I] * 6, None),
    "mrr_segment_any_hit": ([I] * 4, None),
    "mrr_segment_hit_count": ([I] * 4, None),
    "mrr_segment_prefix_sum": ([I, I], I),
    "mrr_segment_scatter": ([I] * 8, None),
    "mrr_bvh_traverse": ([I] * 6 + [F, F, I, F] + [I] * 8, None),
    "mrr_bvh_any_hit": ([I] * 6 + [F, F, I, F] + [I] * 5, None),
}

_USE_MOJO = os.environ.get("MOJORTRE_RAY_BACKEND", "mojo") != "numpy"


class BuildError(RuntimeError):
    pass


def use_mojo() -> bool:
    """Whether the public API dispatches into the Mojo kernels."""
    return _USE_MOJO


def set_backend(name: str) -> None:
    global _USE_MOJO
    if name not in ("mojo", "numpy"):
        raise ValueError(f"unknown backend {name!r}; expected 'mojo' or 'numpy'")
    _USE_MOJO = name == "mojo"


def backend() -> str:
    return "mojo" if _USE_MOJO else "numpy"


def mojo_command() -> list[str]:
    override = os.environ.get("MOJORTRE_RAY_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJORTRE_RAY_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-rtree-ray.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        # A compiler that cannot be launched at all is the same failure as one
        # that fails to compile: the caller cannot use a kernel either way, and
        # `FileNotFoundError` out of a `subprocess` says nothing useful about it.
        raise BuildError(
            f"could not run {' '.join(cmd)}: {exc}"
        ) from exc
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    arr = np.asarray(a, dtype=np.float64)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    elif copy:
        arr = arr.copy()
    return arr


_INT32 = np.iinfo(np.int32)


def i32(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous int32 view of `a`, copying only when it has to.

    Two conversions are refused rather than performed, because the kernels index
    memory with these values and a silently wrong index is a read or a write
    outside the caller's buffer, not a wrong answer:

    - a float array is a caller mistake rather than an index buffer, and
      narrowing it would truncate `1.5` to `1`;
    - an integer that does not fit in int32 -- an unsigned 64-bit id, or an
      int64 index -- would wrap, and `2**31 + 1` becoming a negative offset is
      an out-of-bounds access rather than a detectable error. `np.astype`
      performs that wrap silently, so the range is checked here.
    """
    arr = np.asarray(a)
    if arr.dtype.kind == "f":
        if not np.array_equal(arr, np.floor(arr)):
            raise TypeError(f"index array must hold whole numbers, got dtype {arr.dtype}")
        arr = arr.astype(np.int32)
    else:
        if arr.size and arr.dtype.kind in "iu":
            lo = int(arr.min())
            hi = int(arr.max())
            if lo < _INT32.min or hi > _INT32.max:
                raise OverflowError(
                    f"index array does not fit int32: [{lo}, {hi}] in dtype {arr.dtype}"
                )
        arr = np.asarray(a, dtype=np.int32)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    elif copy:
        arr = arr.copy()
    return arr


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojortreray._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
