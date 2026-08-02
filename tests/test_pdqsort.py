from __future__ import annotations

import ctypes
import random

import numpy as np
import pytest

from pdqsort import PDQSort, pdqsort, sort


def test_python_compatibility_api_sorts_in_place() -> None:
    values = [19, -4, 19, 0, 7, -100, 8, 8, 3]
    result = PDQSort.sort(values)
    assert result is None
    assert values == [-100, -4, 0, 3, 7, 8, 8, 19, 19]

    floats = [3.5, -0.0, 2.25, -8.0, 3.5]
    assert pdqsort(floats) is None
    assert floats == [-8.0, -0.0, 2.25, 3.5, 3.5]


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
def test_all_exported_numeric_types(dtype: np.dtype) -> None:
    source = [91, 0, 7, 7, 3, 88, 2, 45, 1, 9, 9, 4, 63, 5, 6]
    values = np.asarray(source, dtype=dtype)
    expected = np.sort(values.copy())
    sort(values)
    np.testing.assert_array_equal(values, expected)


def test_random_arrays_cross_algorithm_thresholds() -> None:
    random_source = random.Random(0xD05)
    for length in [0, 1, 2, 23, 24, 25, 127, 128, 129, 257, 1024, 4097]:
        values = np.asarray(
            [random_source.randrange(-10_000, 10_001) for _ in range(length)],
            dtype=np.int64,
        )
        expected = np.sort(values.copy())
        sort(values)
        np.testing.assert_array_equal(values, expected)


def test_large_array_uses_parallel_partition_sort() -> None:
    random_source = np.random.default_rng(0xD05)
    values = random_source.integers(-(1 << 62), 1 << 62, 300_000, dtype=np.int64)
    expected = np.sort(values.copy())
    sort(values)
    np.testing.assert_array_equal(values, expected)


@pytest.mark.parametrize(
    "values",
    [
        np.arange(8192, dtype=np.int64),
        np.arange(8191, -1, -1, dtype=np.int64),
        np.zeros(8192, dtype=np.int64),
        np.arange(8192, dtype=np.int64) % 5,
        np.asarray(list(range(4096)) + list(range(4095, -1, -1)), dtype=np.int64),
        np.asarray([value // 64 for value in range(8192)], dtype=np.int64),
    ],
    ids=["sorted", "reverse", "equal", "sawtooth", "organ-pipe", "blocks"],
)
def test_pattern_defeating_inputs(values: np.ndarray) -> None:
    expected = np.sort(values.copy())
    sort(values)
    np.testing.assert_array_equal(values, expected)


def test_noncontiguous_and_non_native_arrays_are_copied_back() -> None:
    backing = np.asarray([99, 8, 98, 3, 97, 7, 96, 1, 95, 2], dtype=">i8")
    view = backing[1::2]
    sort(view)
    np.testing.assert_array_equal(view, np.asarray([1, 2, 3, 7, 8]))
    np.testing.assert_array_equal(backing[::2], np.asarray([99, 98, 97, 96, 95]))


def test_unaligned_arrays_are_copied_before_calling_native_code() -> None:
    storage = np.zeros(1 + 8 * 6, dtype=np.uint8)
    values = storage[1:].view(np.int64)
    values[:] = [5, -1, 8, 0, 3, 2]
    assert not values.flags.aligned

    sort(values)

    np.testing.assert_array_equal(values, [-1, 0, 2, 3, 5, 8])


def test_float_sort_puts_nans_after_ordered_values() -> None:
    values = np.asarray([np.nan, 3.0, -np.inf, np.nan, 1.0, np.inf, -2.0])
    sort(values)
    np.testing.assert_array_equal(
        values,
        np.asarray([-np.inf, -2.0, 1.0, 3.0, np.inf, np.nan, np.nan]),
    )


def test_rejects_arrays_with_overlapping_elements() -> None:
    value = np.asarray([3], dtype=np.int64)
    overlapping = np.lib.stride_tricks.as_strided(value, shape=(3,), strides=(0,))

    with pytest.raises(ValueError, match="overlapping"):
        sort(overlapping)


def test_c_abi_can_be_called_without_python_adapter() -> None:
    from pdqsort import _LIBRARY

    assert isinstance(_LIBRARY, ctypes.PyDLL)

    values = (ctypes.c_int64 * 7)(10, -1, 4, 4, 100, 0, -20)
    address = ctypes.addressof(values)
    assert _LIBRARY.pdqsort_i64(address, len(values)) == 0
    assert list(values) == [-20, -1, 0, 4, 4, 10, 100]
    assert _LIBRARY.pdqsort_i64(0, -1) == -1
    assert _LIBRARY.pdqsort_i64(0, 2) == -1


def test_rejects_inputs_that_cannot_cross_numeric_abi() -> None:
    with pytest.raises(TypeError):
        sort([1, 2.0])
    with pytest.raises(OverflowError):
        sort([0, 1 << 80])
    with pytest.raises(ValueError):
        sort(np.zeros((2, 2), dtype=np.int64))
    with pytest.raises(TypeError):
        sort(np.asarray(["b", "a"]))
