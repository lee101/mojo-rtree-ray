"""mojo-rtree-ray: R-tree backed ray/scene traversal with the inner loops in Mojo.

There is no package called `rtree-ray` on PyPI or conda-forge, so this is a
port of the algorithm that name describes rather than of a single upstream
distribution. What is covered, and what each name is measured against, is
written out in the README.
"""

from ._lib import backend, build, lib, set_backend, use_mojo
from .bvh import BVH, any_hit, traverse, traverse_ranged
from .index import Index, Scene
from .primitive import (
    box_surface_area,
    ray_aabb_any_hit,
    ray_aabb_intersect,
    ray_at,
    ray_make,
    ray_offset,
    ray_sphere_intersect,
    ray_transform,
    ray_triangle_barycentric,
    ray_triangle_intersect,
    triangle_bounds,
)
from .ray import Ray, as_flat
from .traverse import (
    Hit,
    Hits,
    cull_ray_boxes,
    intersect_all,
    intersect_ray,
    intersect_rays,
    occluded,
    occluded_bvh,
    occluded_boxes,
    occluded_rays,
    traverse_bvh,
)

__version__ = "0.1.0"

__all__ = [
    "BVH",
    "Hit",
    "Hits",
    "Index",
    "Ray",
    "Scene",
    "any_hit",
    "as_flat",
    "backend",
    "box_surface_area",
    "build",
    "cull_ray_boxes",
    "intersect_all",
    "intersect_ray",
    "intersect_rays",
    "lib",
    "occluded",
    "occluded_bvh",
    "occluded_boxes",
    "occluded_rays",
    "ray_aabb_any_hit",
    "ray_aabb_intersect",
    "ray_at",
    "ray_make",
    "ray_offset",
    "ray_sphere_intersect",
    "ray_transform",
    "ray_triangle_barycentric",
    "ray_triangle_intersect",
    "set_backend",
    "traverse",
    "traverse_bvh",
    "traverse_ranged",
    "triangle_bounds",
    "use_mojo",
]
