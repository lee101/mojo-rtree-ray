# mojo-rtree-ray

R-tree backed ray/scene traversal with the inner loops written in Mojo.

A scene is a triangle soup. A query is a batch of rays. This library does the
three stages of answering one, in this order: the R-tree turns each ray into a
candidate list, the Mojo kernels test every candidate exactly, and the
surviving answers are reduced to a closest hit, an occlusion flag, or an
ordered list of every hit. The stages that are compute-bound -- the
intersection test and the traversal -- are in Mojo, behind a C ABI reached with
ctypes. The R-tree itself is not rewritten: it is libspatialindex, driven
through the `rtree` package, because it is already C and because the interesting
work in a ray tracer happens after the tree has answered.

## About the name

There is no package called `rtree-ray` on PyPI or on conda-forge, and no
repository of that name on GitHub (checked against the PyPI simple index and
the GitHub repository search; PyPI has `rtree`, `rtreecpy`, `rtree-geodetic`
and `rtreelib`, none of them this). Rather than abandon the target, this ports
the algorithm the name describes and says plainly what each piece is compared
against:

- **the spatial index** is the real `rtree` package (libspatialindex 1.x,
  installed from conda-forge). `mojortreray.Index` is a drop-in stand-in for
  `rtree.index.Index` and `tests/test_index.py` asserts it returns exactly what
  libspatialindex returns for the same operations.
- **the intersection and traversal kernels** have no upstream distribution to
  diff against, so parity rests on two checks that a reader can run.
  `python/mojortreray/reference.py` is a NumPy transcription of the same branch
  structure, written the way these routines are normally written in NumPy, and
  every kernel is tested against it under both backends. Independently of that,
  the traversal is checked against brute force over every triangle in the scene
  -- no structure at all, so nothing about the R-tree's culling or the BVH's
  pruning can hide behind itself. `tests/test_kernels.py` adds a third check that
  shares no code with either: each of the twenty exported kernels is called by
  name and asserted against a value worked out in the test itself.
- **the API shape** follows the two de-facto references for the problem:
  `Ray(origin, direction, tmin, tmax)` with `at`, `offset`, `normalize` and
  `transform` as pyembree defines them, and `Index` with `insert`, `delete`,
  `intersection`, `count`, `nearest`, `contains`, `leaves`, `get_size` and
  `bounds` as `rtree.index.Index` defines them.

## Install

```
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles `src/capi.mojo` into `dist/libmojo-rtree-ray.so`. The
Python wrapper rebuilds it on first use if it is missing or stale, so
`PYTHONPATH=python python -c "import mojortreray"` is enough outside pixi as
long as a `mojo` compiler is on `PATH` (or `MOJORTRE_RAY_MOJO` points at one).
The import itself is free -- the library is loaded on the first call that needs
a kernel -- and with no compiler reachable that call raises `BuildError` with
the compiler's own output rather than an `ImportError`. The package needs
`numpy` and `rtree` at runtime either way.

## Usage

```python
import numpy as np
import mojortreray as mrr

# A quad grid: 8x8x8 cells, two triangles each.
scene = mrr.Scene.grid(shape=(8, 8, 8), seed=1)
print(scene.ntriangles, "triangles")

# A batch of camera rays, in the flat (n, 6) layout the kernels read.
lo, hi = scene.bounds
rng = np.random.default_rng(0)
origin = np.column_stack([np.full(1000, lo[0] - 0.5),
                          rng.uniform(lo[1], hi[1], 1000),
                          rng.uniform(lo[2], hi[2], 1000)])
target = rng.uniform(lo, hi, (1000, 3))
direction = target - origin
direction /= np.linalg.norm(direction, axis=1)[:, None]
rays = np.ascontiguousarray(np.concatenate([origin, direction], axis=1))

# Closest hit per ray. `Hit.point`, `.normal` and `.uv` are computed on demand.
hits = mrr.intersect_rays(rays, scene)
first = hits[0]
if first is not None:
    print("hit triangle", first.prim, "at t =", first.t)
    print("point", first.point, "normal", first.normal)

# Every hit, grouped by ray and sorted by distance within a ray.
all_hits = mrr.intersect_all(rays, scene)
print(len(all_hits), "hits over", len(rays), "rays")

# Occlusion, and the single-ray and BVH forms.
print("blocked:", mrr.occluded_rays(rays, scene)[:8])
bvh = scene.bvh()
print("blocked via bvh:", mrr.occluded_bvh(rays, bvh, scene.soup)[:8])
t, prim, order, ocount = mrr.traverse_bvh(rays, bvh, scene.soup)
print("closest via bvh:", t[:4], prim[:4])

# One ray on its own, with its own interval.
ray = mrr.Ray([0.0, 0.0, 5.0], [0.0, 0.0, -1.0], 0.0, 100.0)
hit = mrr.intersect_ray(ray, scene)
print("single ray:", hit)
```

The single-primitive routines are usable on their own, and take the same
argument order as the kernels:

```python
mrr.ray_triangle_intersect(v0, v1, v2, origin, direction,
                           tmin=0.0, tmax=np.inf, cull_backface=False, eps=1e-9)
mrr.ray_aabb_intersect(origin, direction, lo, hi, tmin=0.0, tmax=np.inf, eps=0.0)
mrr.ray_aabb_any_hit(origin, direction, lo, hi, tmin=0.0, tmax=np.inf)
mrr.ray_sphere_intersect(origin, direction, center, radius, tmin=0.0, tmax=np.inf)
mrr.ray_triangle_barycentric(v0, v1, v2, origin, direction, eps=1e-9)
mrr.triangle_bounds(vertices, faces)
mrr.box_surface_area(lo, hi)
mrr.cull_ray_boxes(ray, boxes)
mrr.occluded_boxes(rays, boxes)
```

`mrr.set_backend("numpy")` switches the whole public API to the NumPy reference.
The two are required to agree, and `tests/` is what holds them to it.

## What is covered

Everything below is in `src/mojortreray/`, in this order, which is the order the
file reads.

| module | routine | what it is |
| --- | --- | --- |
| `ray.mojo` | `ray_make` | `Ray.normalize` |
| | `ray_at` | `Ray.at` |
| | `ray_offset` | `Ray.offset` |
| | `ray_transform` | `Ray.transform`, row-major 4x4 |
| `primitive.mojo` | `SlabRay` | one ray's reciprocals, hoisted out of the box loop |
| | `ray_aabb_intersect` | slab test, entry parameter or miss |
| | `ray_aabb_any_hit` | the same slabs, short-circuiting |
| | `ray_sphere_intersect` | quadratic in the ray parameter |
| | `ray_triangle_intersect` | Moller-Trumbore, `Fast MinimumStorage` form |
| | `ray_triangle_barycentric` | `(u, v)` of an accepted hit |
| | `triangle_bounds` | per-triangle AABB of an indexed mesh |
| | `box_surface_area` | the SAH cost estimate |
| `traverse.mojo` | `ray_triangle_intersect_batch` | every `(ray, candidate)` pair |
| | `ray_aabb_intersect_batch` | every `(ray, node)` pair |
| | `segment_closest_hit` | reduce a ray's candidates to one |
| | `segment_any_hit` | occlusion over a candidate list |
| | `segment_hit_count` | counts for the counting sort |
| | `segment_prefix_sum` | in place, returns the total |
| | `segment_scatter` | place each ray's hits contiguously |
| `bvh.mojo` | `bvh_traverse` | closest-hit descent, with the visit order |
| | `bvh_any_hit` | any-hit descent, stops at the first hit |

The R-tree stage and the `Scene` bookkeeping around it are Python and
libspatialindex, not Mojo; `mojortreray.Index` mirrors `rtree.index.Index` and
`mojortreray.bvh.BVH.build` is a median-split builder in NumPy.

The R-tree stage itself is one C call: `Scene.candidate_pairs` builds every
query box with `Scene.query_boxes` and issues a single bulk
`Index.intersection_v`. `Index.intersection_v` is the passthrough for
`rtree.index.Index.intersection_v`; `Index.intersection` is untouched and still
the per-box generator, so the wrapper remains a drop-in for the real thing.

## What is not covered, and why

- **The R-tree is not ported.** `Index` delegates to libspatialindex. A spatial
  index is a C data structure with a decades-old tuning story, and rewriting it
  in Mojo would have been neither the compute-bound part of the problem nor an
  improvement.
- **Hit points, normals and barycentrics are computed in Python, not Mojo.**
  They are evaluated once per surviving hit, not once per candidate, so they are
  not compute-bound. `Hit` computes them lazily and caches them.
- **`intersect_all` sorts in NumPy.** The Mojo kernels do the counting sort and
  the scatter; the final ordering within a ray is an `np.lexsort` over the flat
  hit list, because a per-ray sort inside a flat batch is not worth a kernel of
  its own at the hit counts a scene produces.
- **`traverse_ranged` splits the batch in Python.** The kernel takes scalar
  `tmin`/`tmax`; a batch with a different interval per ray is partitioned into
  maximal runs of equal bounds and the kernel is called once per run. Per-ray
  bounds would need an extra `(n, 2)` buffer and a second pass over the slab
  test for no arithmetic saving.
- **The segment reductions are serial.** `segment_closest_hit`,
  `segment_any_hit`, `segment_hit_count` and `segment_scatter` divide cleanly
  over rays, and splitting them with `max.algorithm.parallelize` was
  implemented and then reverted: it measured slower than the serial loop at
  every worker count tried. They are single streaming passes that are already
  bandwidth-bound, and the launch costs more than the work. The compute-bound
  batched kernels, which do gain from the split, keep it.
- **Under the `numpy` backend, `bvh.any_hit` reports the closest hit as the
  blocker.** There is no separate any-hit descent in Python; the boolean answer
  is the same, and the primitive id is a valid blocker but not necessarily the
  one the Mojo kernel stops on.
- **The BVH builder is a plain median split, not SAH.** A surface-area sweep
  would produce a better tree at build time; the split is the cheapest one that
  still gives a usable hierarchy, and a better build belongs in the same pass
  that would then measure it.
- **A long camera ray is a poor query for an R-tree.** See the note on the
  end-to-end rows below: the ray's bounding box over an unbounded interval
  covers a large fraction of a dense scene, so the R-tree returns a large
  candidate list. The candidate list is still a strict superset of the true hits
  -- over-selection costs time and under-selection would be a correctness bug --
  but on dense volumetric scenes the BVH is the path to use.

## Benchmarks

Machine: Linux x86_64, 72 logical CPUs, Intel(R) Xeon(R) CPU E5-2697 v4 @
2.30GHz. NumPy 2.5.1, Mojo 1.2.0.dev2026092905, `rtree` 1.4.1
(libspatialindex). Best of N runs, via `pixi run bench` (the task holds a
machine-wide lock so concurrent factory jobs cannot distort the numbers).

The table below is the output of one `pixi run bench` on that machine, printed
by the benchmark itself. Read the ratios as order-of-magnitude claims: this is a
shared machine and the numbers move between runs with no code change. Across
four runs of this benchmark the triangle batch row measured 43.9, 34.7, 34.1 and
52.4 ms (37x, 46x, 47x and 31x), the AABB batch 33.5, 35.4, 43.1 and 34.1 ms
(36x, 33x, 28x and 35x), and the end-to-end `intersect_rays` row 3.60x, 3.37x,
3.40x and 3.25x. The first two rows are the noisiest; the BVH and end-to-end
rows are stable to within about 15%. The kernels win by tens of times on every
run, which is the claim the table supports.

| kernel | mojo-rtree-ray | numpy reference | ratio | note |
| --- | ---: | ---: | ---: | --- |
| ray_triangle_intersect_batch (4M pairs, 50k rays x 80 candidates) | 52.4 ms | 1625.3 ms | 31.02x |  |
| ray_aabb_intersect_batch (4M pairs, 20k rays x 200 nodes) | 34.1 ms | 1182.6 ms | 34.71x |  |
| BVH closest-hit vs NumPy brute force (3k rays, 8192 triangles) | 8.7 ms | 10127.2 ms | 1161.06x |  |
| BVH occlusion vs NumPy brute force (3k rays, 8192 triangles) | 10.4 ms | 9799.4 ms | 945.30x |  |
| BVH closest-hit throughput (200k rays, 65536 triangles) | 57.4 ms | not run | - | brute force would be 13G pairs |
| intersect_rays end to end (3k rays, 3456 triangles) | 734.3 ms | 2388.7 ms | 3.25x |  |
| intersect_all end to end (3k rays, 3456 triangles) | 739.5 ms | 2411.1 ms | 3.26x |  |
| R-tree cull for 3k rays, 3456 triangles | 662.0 ms | not run | - | libspatialindex; this is the bound on the whole query |
| R-tree cull, wrapper vs rtree (2k rays, 3456 triangles) | 1215.5 ms | 1200.4 ms | 0.99x | both sides are libspatialindex |

Reading the table honestly:

- **The kernels are roughly 30x to 47x faster than the NumPy reference** on the
  candidate-list work, and the traversal is three orders of magnitude faster
  than the brute-force baseline it replaces. No Mojo kernel is slower than its
  baseline.
- **The end-to-end rows are about 3.3x** (3.60x, 3.37x, 3.40x and 3.25x across the
  four runs). That number is small next to the kernel rows because the R-tree
  cull dominates the query, and the cull is
  libspatialindex in C, which no amount of Mojo reaches. The cull row and the
  `intersect_rays` row are the same measurement within run-to-run noise.
  Measured directly on the same 3k-ray batch, twice: `Scene.candidate_pairs`
  totals 649.1 ms and 661.5 ms, of which `Index.intersection_v` -- the
  libspatialindex descent itself -- is 645.4 ms and 655.9 ms, and
  `Scene.query_boxes` is 0.4 ms both times. So about 99% of the end-to-end cost
  is third-party C, and the Python around it is already a rounding error.
- **The two BVH rows compare against a deliberately weak baseline.** It
  materialises an `(n * ntri, 3, 3)` gather in NumPy, so it is memory-bound as
  well as algorithmically naive. It is the right baseline for "what an
  unaccelerated traversal costs" and the wrong one for "what a good NumPy
  implementation costs", and the row is labelled accordingly.
- **The R-tree wrapper row sits on 1.0 and that is the honest answer.** It
  measures a Python method call in front of libspatialindex, which forwards
  straight through and is untouched by any of the work here. A few percent on a
  call that only delegates is the noise floor of this machine. This row has come
  out at 1.07x, 1.05x, 0.97x and 0.99x across four runs, so it is 1.0 plus or
  minus the noise, and the benchmark flags the runs below 1.0 rather than
  hiding them.
- **The machine is shared and the small rows are noisy.** `pixi run bench`
  holds a machine-wide lock, but other work still lands, and the lock itself is
  contended -- during this pass a `pixi run bench` waited two hours behind
  another repo's benchmark. Read the kernel rows as "tens of times faster",
  not as a precise ratio.

The per-kernel effect of the hoisted slab reciprocals is not readable from the
table above, because this benchmark is noisy at the few-millisecond scale: on
this machine the same row has moved by a factor of two between runs of
`pixi run bench` with no code change in between. No before/after timing for that
change is quoted here, because the pair of shared libraries it needs is not in
the repository and the numbers cannot be regenerated or checked by a reader.
What *is* checked is the property the change rests on, which is that hoisting
the reciprocal cannot move the numerics: `tests/test_traverse.py` pins the
batched form against the single-box `ray_aabb_intersect` box-for-box and against
the NumPy reference for general, axis-parallel and fully degenerate directions,
at more than one worker count, and pins the BVH against brute force for the same
rays.

## Notes on the shape of the code

Two decisions are worth explaining, because both are visible in the source and
neither is obvious from the API.

### The cull is batched through `intersection_v`

`Scene.candidate_pairs` issues a single bulk `Index.intersection_v` for the
whole batch rather than one `Index.intersection` generator per ray.
`intersection` is a Python generator, so the per-ray form costs a ctypes round
trip into libspatialindex and another per result it yields;
`intersection_v` -- `rtree`'s own bulk form, which needs `rtree >= 1.1` -- moves
the loop into C and returns the flat ids grouped by box plus a per-box count.
`Scene.query_boxes` is the matching vectorised box pass: the same arithmetic as
the scalar `query_box`, done on all three axes for all rays at once.

The result is identical, and that is asserted rather than asserted-in-prose:
`tests/test_traverse.py` pins the batched form against the per-ray
`Scene.candidates` it replaced -- the ids, the segment offsets, and the order
inside each segment -- for four different `tmin`/`tmax` settings, on a live
batch, a batch that all misses, a mixed batch, and an empty one. It also pins
the vectorised `query_boxes` against scalar `query_box` over the same settings,
including the axis-parallel (`inf * 0.0` is a NaN) and `tmin == tmax` cases.

After this the stage is essentially all libspatialindex, which is why the cull
row and the end-to-end row measure the same thing.

### What the slab test hoists

The slab test is the one place a batched caller had real redundant work. It
divides by `direction[a]` on every axis of every box, but the direction is fixed
for the whole ray: the AABB batch applies one ray to every candidate node, and
the BVH applies one ray to every node it visits. `SlabRay` in
`src/mojortreray/primitive.mojo` builds the three reciprocals and the three
"is this axis parallel" flags once per ray; the inner test then just multiplies.

Everything downstream of `1.0 / d` is unchanged, so the results are identical
to the per-box form -- which is asserted, not assumed (see the tests named above;
this benchmark is too noisy at the millisecond scale to resolve a change that
size). The single-box `ray_aabb_intersect` is now a one-line wrapper over the
same code, so there is one implementation of the slab arithmetic rather than
two.

The `par_*` branch is the reason the flags are hoisted alongside the reciprocals
rather than recomputed. A ray travelling exactly along an axis would, under the
usual `1.0 / 0.0` trick, multiply an infinite reciprocal by a zero offset and
get a NaN; a NaN fails every comparison and the box is reported as a miss
precisely when the ray is inside that slab. Branching on the zero direction is
what keeps that case a hit, and hoisting the branch keeps it exact.
`tests/test_traverse.py` pins the batched form against the single-box test and
against the reference for parallel and fully degenerate directions, at 1 and 2
workers, and pins the BVH against brute force for the same rays.

### Why the batch kernels are not vectorised

This is the honest result of walking the escalation ladder, and it is a
measurement rather than an assumption.

The two batch kernels are the obvious SIMD candidates, and they are left scalar
on purpose. Each output element tests a *different* triangle or box at a
data-dependent address -- `tri[k]` or `box_idx[k]` indexes into the soup -- so a
vector across `k` needs a gather, not a contiguous load, and a gather that reads
9 doubles per lane does not buy anything over 9 scalar loads.

More to the point, the kernels are not compute-bound to begin with. The triangle
batch moves about 72 bytes of soup per pair plus the outputs: 4M pairs is
roughly 290 MB, which at 72 workers takes ~20 ms, or about 17 GB/s. That is a
memory-bandwidth ceiling, not a floating-point one, and a wider vector moves no
bytes any faster. The kernels are already tens of times ahead of a NumPy baseline
that *is* fully vectorised and multithreaded; there is no arithmetic headroom
left to widen into. Widening a bandwidth-bound loop is the classic way to end up
with more instructions and the same wall clock.

That the emitted code is scalar is checkable rather than asserted: the built
shared library contains about 265 scalar `*sd` floating-point instructions and 5
packed `*pd` ones, and `objdump -d dist/libmojo-rtree-ray.so` is the check.

### The segment reductions are serial

`segment_closest_hit`, `segment_any_hit`, `segment_hit_count` and
`segment_scatter` divide cleanly over rays, and were tried split with
`max.algorithm.parallelize`. They are single streaming passes that are already
bandwidth-bound, so the split lost at every worker count tried: a thread launch
costs more than the roughly 20 microseconds of work each thread gets at the
batch sizes real callers pass. They are serial in the source and say so. The two
compute-bound batched kernels, which do gain from the split, keep theirs behind
a pair-count threshold.

The threshold matters. The kernels take `nray * 32 >= 16384` as the condition
for going parallel, and the Python callers pass `os.cpu_count()` as the worker
count, so a small batch still enters the parallel branch's test and falls
through to the serial loop. Both branches write the output in pair order, so the
two paths are byte-identical, and
`tests/test_traverse.py::test_batch_and_worker_counts_agree` pins that at 1, 2
and 4 workers.

### Why there is no GPU path

Not because the API is unavailable -- `max.gpu.host.DeviceContext` and
`max.gpu` both compile on this toolchain. There is simply no kernel here worth
moving:

- the triangle batch is about 30 flops per pair against roughly 80 bytes of
  triangle soup plus the shared ray, which is under 0.5 flops per byte. The
  AABB batch is lower still. Both are far below the roughly 2 flops per byte
  where a GPU wins, and both are already more than 30x ahead of a fully
  vectorised baseline on the CPU.
- the BVH descent is pointer-chasing with a stack, dominated by cache misses on
  node bounds -- the case a GPU handles worst.
- the stage that actually costs the wall clock, the R-tree cull, is
  libspatialindex in C, and is not this project's to port.

Shipping a GPU path here would be a launch-and-copy cost added to a loop that
already runs in single-digit milliseconds. The API is available and was checked;
the arithmetic is not.

## How it works

### FFI

`src/capi.mojo` is the only compilation unit, built with
`mojo build --emit shared-lib -I src src/capi.mojo`. It holds 20 `@export`
entry points whose bodies are pure plumbing: rebuild the pointer, forward to the
kernel in the module that owns it.

Every buffer crosses the ABI as an `Int` address, because `@export` rejects
parametric functions and a pointer with an inferred origin is parametric. The
body rebuilds it with
`Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=addr)`. Nothing on
the Mojo side allocates; all memory, including scratch, belongs to the caller.
That is why the kernels take `stack`, `order`, `ocount` and the reduction
scratch as arguments.

`@export` requires an explicit `abi("C")` effect and an `abi("C")` function may
not be `raises`, so no kernel does anything that can fail.

On the Python side `python/mojortreray/_lib.py` declares every signature once,
loads the shared object lazily, and rebuilds it if any `.mojo` file is newer.
`f64`/`i32` hand out C-contiguous views, copying only when they have to, and
`addr` is just `array.ctypes.data`. A ray batch is packed to C order before it
crosses: the kernels address ray `i` at element `6 * i`, so a transposed view
would otherwise be read with the wrong stride and answer with another ray's
result. An index array holding anything other than whole numbers is rejected
rather than narrowed.

### Memory layout

Everything is float64 and 8-byte aligned, with 32-bit `int32` for indices.

| array | shape | meaning |
| --- | --- | --- |
| `rays` | `(n, 6)` | `origin.xyz direction.xyz`, one ray per row |
| `soup` | `(ntri, 9)` | three triangle vertices, flat, no index indirection |
| `pair_tri` | `(npairs,)` int32 | primitive id of each `(ray, candidate)` pair |
| `offsets` | `(n + 1,)` int32 | ray `i` owns `offsets[i]:offsets[i + 1]` |
| `t`, `hit` | `(npairs,)` | the per-pair result; `hit` is a 0.0/1.0 mask |
| `boxes` | `(k, 6)` | `lo.xyz hi.xyz` |
| `nbox` | `(nodes, 6)` | BVH node bounds |
| `nint` | `(nodes, 4)` int32 | `left, right, first, count`; `count > 0` leaf, `== 0` interior, `< 0` empty |

A mesh with shared vertices is de-indexed into `soup` once, by `Scene`. The
traversal kernels then index a triangle with one multiply and no gather, which
is the reason `triangle_bounds` still takes the indexed `(vertices, faces)`
form and is the only kernel that reads vertex indices.

The candidate list is the load-bearing structure. Once the R-tree has answered,
every remaining stage -- test, reduce, scatter -- is a straight-line loop over
the flat pair list, and the per-ray segment is expressed purely as a range of
that list. Nothing in the kernel has to know what a scene is.

### Two conventions worth stating

`MISS` is `-1.0`, and a hit is never negative because `tmin` is clamped. An
origin inside the geometry hits at `tmin`, not at a negative parameter.

The slab test branches on a zero direction component rather than dividing by it.
`1.0 / 0.0` is an infinity and `(0 - 0) * inf` is a NaN, and a NaN fails every
comparison -- so the usual shortcut reports a miss precisely for the ray that is
traveling inside that slab. The NumPy reference does the same thing, for the
same reason, and `tests/test_primitive.py` asserts the axis-parallel case.

## Tests

```
pixi run test
```

126 tests in four files. `test_kernels.py` calls all twenty exported kernels by
name through ctypes and checks each against a value worked out in the test
rather than by re-running the same wrapper, so a kernel that is mis-wired
behind the public API cannot pass; it also asserts that the set of `@export`
names, the set of names in `_lib._SIGNATURES` and the set this file calls are
the same set, so a kernel added without a test fails the build. It is also what
found the three FFI defects listed below. `test_primitive.py` compares every
single-primitive routine against the reference under both backends and pins the
edge cases that are easy to get wrong: an origin inside the box, an interval that
starts past the box exit, a direction with a zero component, a degenerate
triangle, one-sided winding. `test_traverse.py` compares the batched kernels,
the reductions, the BVH descent and the whole pipeline, and asserts the property
that actually matters -- the closest hit found through the R-tree, the closest
hit found through the BVH, and the closest hit found by testing every triangle
are the same distance, and the reported primitive is one of the triangles that
reach it. Because a scene of quads split into two triangles has coplanar pairs,
that last assertion is deliberately tie-tolerant: a ray through the shared
diagonal hits both triangles at the same place to within a rounding error, so
which one is reported is a tie-break rather than an error. The tests also pin
the batched cull against the per-ray form it replaced: the vectorised
`query_boxes` against scalar `query_box` for four `tmin`/`tmax` settings
including the axis-parallel and `tmin == tmax` cases, and `candidate_pairs`
against `Scene.candidates` for the ids, the segment offsets and the order inside
each segment, on a live batch, a batch that all misses, a mixed batch, and an
empty one. `test_index.py` compares `Index` against a real
`rtree.index.Index` for every method, and asserts that `Scene.candidates`
never omits a triangle that actually intersects the ray.

The hoisted slab reciprocals are pinned against the form they replaced: the
batched AABB test has to agree box-for-box with the single-box
`ray_aabb_intersect` for general, axis-parallel and fully degenerate directions,
and the batched and BVH forms have to agree with the NumPy reference at more
than one worker count. Those three tests are what makes "the output is bitwise
identical" a checked claim.

The FFI boundary is where this design is most exposed, and each of the ways it
can go wrong has its own test:

- a transposed (non-C-contiguous) ray batch has to answer exactly as the packed
  one does, because the kernels address ray `i` at element `6 * i`;
- a `Hit` has to keep resolving `point` and `uv` against the batch it was found
  in, after the scene has been queried again with a different one;
- a fractional index array has to be rejected rather than truncated, and so does
  an integer that does not fit in the int32 the kernels address memory with --
  `np.astype` wraps it silently, and `2**31 + 1` becoming a negative offset is an
  out-of-bounds access rather than a detectable error;
- a face index outside the vertex array has to be rejected before the kernel is
  entered. `triangle_bounds` indexes `vertices` unchecked, so a face index one
  element past the end reads heap memory and a large one segfaults; this was a
  real crash, found by probing the boundary rather than by reading the code.

Two more defects were found the same way, and both are now pinned. `BVH.load`
returned an `NpzFile` without closing it, leaking a file descriptor on every
call, and the traversal of a BVH over no primitives never terminated: an all-zero
node row reads as an interior node whose two children are itself, so each push
re-entered the same node. A node with `count < 0` is now an explicitly empty
node, and a node whose child index is itself is skipped, so termination does not
rest on the builder having produced a well-formed tree.

The suite is not vacuous: dropping the `u + v > 1` far-edge constraint from the
Mojo triangle test -- a one-token change -- fails 12 of the 126 tests, spread
across the primitive, batch, BVH and end-to-end levels.

## Layout

```
src/capi.mojo                 the C ABI: 20 exported entry points
src/mojortreray/ray.mojo      ray algebra
src/mojortreray/common.mojo    shared aliases, the Vec3 type, MISS/FAR
src/mojortreray/primitive.mojo  ray/primitive intersection
src/mojortreray/traverse.mojo   batched candidate-list queries
src/mojortreray/bvh.mojo      flat BVH traversal
MOJO_NOTES.md                 the Mojo dialect notes for this toolchain
build/build.sh                mojo build --emit shared-lib
python/mojortreray/_lib.py     ctypes loader, builds on demand
python/mojortreray/ray.py      the Ray value type
python/mojortreray/primitive.py public intersection API
python/mojortreray/bvh.py      BVH type, builder, traversal wrapper
python/mojortreray/index.py    Index (rtree-compatible) and Scene
python/mojortreray/traverse.py the query API: Hit, Hits, intersect_*, occluded_*
python/mojortreray/reference.py the NumPy baseline and the numpy backend
tests/                        pytest: every kernel by name, plus parity
bench/bench.py                benchmarks, prints a markdown table
```

## Licence

MIT. See `LICENSE`.
