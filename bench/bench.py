from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from pdqsort import sort


T = TypeVar("T")


def _median_seconds(
    setup: Callable[[], T],
    operation: Callable[[T], None],
    validate: Callable[[T], None],
    repeats: int,
) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        values = setup()
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            started = time.perf_counter_ns()
            operation(values)
            elapsed = time.perf_counter_ns() - started
        finally:
            if gc_was_enabled:
                gc.enable()
        validate(values)
        samples.append(elapsed * 1e-9)
    return statistics.median(samples)


def _array_case(name: str, source: np.ndarray, repeats: int) -> None:
    expected = np.sort(source.copy())

    def mojo_sort(values: np.ndarray) -> None:
        sort(values)

    def numpy_sort(values: np.ndarray) -> None:
        values.sort(kind="quicksort")

    def validate(values: np.ndarray) -> None:
        if not np.array_equal(values, expected, equal_nan=True):
            raise AssertionError(f"{name} produced an incorrect result")

    _median_seconds(source.copy, mojo_sort, validate, 1)
    _median_seconds(source.copy, numpy_sort, validate, 1)
    mojo = _median_seconds(source.copy, mojo_sort, validate, repeats)
    python = _median_seconds(source.copy, numpy_sort, validate, repeats)
    ratio = python / mojo
    result = "faster" if ratio >= 1.0 else "slower"
    factor = ratio if ratio >= 1.0 else 1.0 / ratio
    print(
        f"{name:24} Mojo {mojo * 1e3:9.3f} ms  "
        f"NumPy {python * 1e3:9.3f} ms  {factor:5.2f}x {result}"
    )


def _list_case(source: list[int], repeats: int) -> None:
    expected = sorted(source)

    def mojo_sort(values: list[int]) -> None:
        sort(values)

    def python_sort(values: list[int]) -> None:
        values.sort()

    def validate(values: list[int]) -> None:
        if values != expected:
            raise AssertionError("Python list benchmark produced an incorrect result")

    _median_seconds(source.copy, mojo_sort, validate, 1)
    _median_seconds(source.copy, python_sort, validate, 1)
    mojo = _median_seconds(source.copy, mojo_sort, validate, repeats)
    python = _median_seconds(source.copy, python_sort, validate, repeats)
    ratio = python / mojo
    result = "faster" if ratio >= 1.0 else "slower"
    factor = ratio if ratio >= 1.0 else 1.0 / ratio
    print(
        f"{'Python list / random':24} Mojo {mojo * 1e3:9.3f} ms  "
        f"Python {python * 1e3:8.3f} ms  {factor:5.2f}x {result}"
    )


def main() -> None:
    rng = np.random.default_rng(0xD05)
    random_small = rng.integers(-(1 << 62), 1 << 62, 10_000, dtype=np.int64)
    random_large = rng.integers(-(1 << 62), 1 << 62, 1_000_000, dtype=np.int64)
    duplicate_large = rng.integers(0, 32, 1_000_000, dtype=np.int64)
    sorted_large = np.arange(1_000_000, dtype=np.int64)

    print("Median sort-call time; input copies and result checks are outside the timer.")
    _array_case("ndarray / random 10k", random_small, 9)
    _array_case("ndarray / random 1m", random_large, 5)
    _array_case("ndarray / 32 values 1m", duplicate_large, 5)
    _array_case("ndarray / sorted 1m", sorted_large, 7)
    _list_case(random_small.tolist(), 7)


if __name__ == "__main__":
    main()
