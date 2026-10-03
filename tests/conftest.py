import numpy as np
import pytest

import mojortreray as mrr


@pytest.fixture(scope="session", autouse=True)
def mojo_backend():
    """Every test runs against the Mojo kernels unless it says otherwise."""
    mrr.set_backend("mojo")
    yield
    mrr.set_backend("mojo")


@pytest.fixture(scope="session")
def scene():
    return mrr.Scene.grid(shape=(6, 6, 6), seed=7)


@pytest.fixture(scope="session")
def rays():
    """A camera looking at the scene plus a set of deliberately awkward rays."""
    grid = []
    for u in range(9):
        for v in range(9):
            d = np.array([0.4, 0.2, 0.1]) + np.array([u, v, 0.0]) * 0.11
            grid.append(np.concatenate([[-2.0, 0.2 * u, 0.2 * v], d / np.linalg.norm(d)]))
    out = np.asarray(grid, dtype=np.float64)
    # rays that miss, that start inside, and that run exactly along an axis
    extra = np.array(
        [
            [100.0, 100.0, 100.0, 1.0, 0.0, 0.0],
            [2.5, 2.5, -5.0, 0.0, 0.0, 1.0],
            [-5.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -5.0, 0.0, 0.0, 1.0],
        ]
    )
    return np.concatenate([out, extra])


def both_backends(fn, *args, **kwargs):
    """Call `fn` under each backend and return the two results."""
    mrr.set_backend("mojo")
    a = fn(*args, **kwargs)
    mrr.set_backend("numpy")
    b = fn(*args, **kwargs)
    mrr.set_backend("mojo")
    return a, b
