# mojo-pdqsort

An in-place, unstable ascending sort for fixed-width numeric data.  The sorting
core is written in Mojo and exposed through a small C ABI and a Python adapter.
It uses pattern-defeating quicksort-style partitioning, insertion sort for small
ranges, and heap sort after repeatedly unbalanced partitions.  Inputs of at
least 262,144 elements may be split into four independent ranges when the first
partitions are sufficiently balanced.

This package supports `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
`uint32`, `uint64`, `float32`, and `float64`.  Floating-point NaNs sort after
all ordered values.  The sort is not stable, and no ordering between equivalent
values (including signed zeroes) is promised.

## Python API

```python
import numpy as np
from pdqsort import sort

values = np.array([4, -2, 9, 4], dtype=np.int64)
sort(values)
assert values.tolist() == [-2, 4, 4, 9]
```

`sort`, its alias `pdqsort`, and `PDQSort.sort` mutate their argument and return
`None`.

Accepted inputs are:

- writable, one-dimensional NumPy arrays with one of the dtypes above; and
- Python lists containing only booleans, only non-boolean integers, or only
  floats/NumPy floating scalars.

Native-endian, C-contiguous, aligned arrays are sorted directly.  Other valid
views are copied to an aligned native-endian buffer and copied back.  Views
whose logical elements overlap are rejected because an in-place sorted result
cannot be represented reliably.  Python integers must fit in signed 64 bits;
integer lists and floating-point lists are sorted through `int64` and `float64`
buffers respectively.

The Python adapter keeps the GIL while Mojo holds an ndarray's data pointer, so
another Python thread cannot resize the array and invalidate that borrowed
pointer during the call.

## Build and test

The checked-in Pixi environment currently targets Linux x86-64 and a pinned
Mojo nightly:

```console
pixi run build
pixi run test
```

The build writes `build/libpdqsort.so` and copies the public header to
`build/pdqsort.h`.  Set `PDQSORT_LIBRARY` to load a shared library from another
location.

The raw C entry points accept an address and an element count and return zero
on success.  They reject negative counts and a null address when more than one
element is requested.  They cannot verify the allocation behind a non-null raw
address: C callers must provide a suitably aligned contiguous buffer containing
at least `length` elements and keep it alive and exclusively mutable until the
call returns.

## Benchmark

Run the reproducible comparison with:

```console
pixi run bench
```

The benchmark reports median wall-clock time for the public Python `sort` call.
For ndarrays it compares against NumPy's in-place `quicksort`; for a Python list
it compares against `list.sort`.  Input copies and correctness checks happen
outside the timed region, and every timed result is checked.  The list result
is an end-to-end API measurement, so the pdqsort time includes conversion to
and from an `int64` NumPy buffer.  These numbers describe the included seeded
integer workloads on the machine running the command; they are not a general
claim that one implementation is always faster.

Measured on 2026-08-26 with the repository's locked benchmark task:

| Workload | Mojo before | Mojo after | Reference after | After comparison |
|---|---:|---:|---:|---:|
| ndarray / random 10k | 0.377 ms | 0.382 ms | NumPy 0.166 ms | 2.30x slower |
| ndarray / random 1m | 50.251 ms | 51.914 ms | NumPy 23.450 ms | 2.21x slower |
| ndarray / 32 values 1m | 9.543 ms | 9.599 ms | NumPy 5.229 ms | 1.84x slower |
| ndarray / sorted 1m | 1.030 ms | 1.357 ms | NumPy 23.693 ms | 17.46x faster |
| Python list / random | 8.854 ms | 1.721 ms | Python 2.405 ms | 1.40x faster |

The optimized list path uses exact built-in type checks, NumPy's native list
conversion, and a bulk `tolist()` copy-back. Native, contiguous ndarrays remain
zero-copy across the Python/Mojo boundary. Long one-sided partition scans use
SIMD after a short scalar probe and finish with a scalar tail; the scalar probe
avoids vector setup on the common early-exit path.

The sort is comparison- and memory-bound, with irregular swaps and well under
two arithmetic operations per byte moved, so it does not justify a GPU path.
No GPU memory was allocated or benchmark sweep run. The pinned toolchain's
`parallelize` runtime was also not retained: worker dispatch from the shared
C-ABI library was not runtime-safe in testing. The size threshold still avoids
large-range decomposition below 262,144 elements, and the independent ranges
are processed serially.
