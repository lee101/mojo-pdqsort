"""In-place pattern-defeating quicksort backed by Mojo.

``sort`` accepts one-dimensional NumPy arrays and homogeneous Python lists of
integers or floats.  NumPy integer dtypes and float32/float64 are sorted without
changing dtype.  Like ``list.sort``, the function mutates its argument and
returns ``None``.
"""

from __future__ import annotations

import ctypes
import os
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["PDQSort", "pdqsort", "sort"]


_DTYPE_SYMBOLS = {
    np.dtype("int8"): "pdqsort_i8",
    np.dtype("int16"): "pdqsort_i16",
    np.dtype("int32"): "pdqsort_i32",
    np.dtype("int64"): "pdqsort_i64",
    np.dtype("uint8"): "pdqsort_u8",
    np.dtype("uint16"): "pdqsort_u16",
    np.dtype("uint32"): "pdqsort_u32",
    np.dtype("uint64"): "pdqsort_u64",
    np.dtype("float32"): "pdqsort_f32",
    np.dtype("float64"): "pdqsort_f64",
}


def _library_path() -> Path:
    configured = os.environ.get("PDQSORT_LIBRARY")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "build" / "libpdqsort.so"


def _load_library() -> ctypes.PyDLL:
    path = _library_path()
    if not path.is_file():
        raise ImportError(f"pdqsort shared library not found at {path}; run `pixi run build`")
    # Keep the GIL while the native function holds a pointer borrowed from an
    # ndarray.  Otherwise another thread can resize the array and free it.
    library = ctypes.PyDLL(str(path))
    for symbol in _DTYPE_SYMBOLS.values():
        function = getattr(library, symbol)
        function.argtypes = (ctypes.c_ssize_t, ctypes.c_ssize_t)
        function.restype = ctypes.c_ssize_t
    return library


_LIBRARY = _load_library()


def _sort_array(values: np.ndarray[Any, Any]) -> None:
    if values.ndim != 1:
        raise ValueError("pdqsort only accepts one-dimensional arrays")
    if not values.flags.writeable:
        raise ValueError("pdqsort requires a writable array")

    native_dtype = values.dtype.newbyteorder("=")
    symbol = _DTYPE_SYMBOLS.get(native_dtype)
    if symbol is None:
        supported = ", ".join(str(dtype) for dtype in _DTYPE_SYMBOLS)
        raise TypeError(f"unsupported dtype {values.dtype}; expected one of: {supported}")
    if values.size > 1 and abs(values.strides[0]) < values.itemsize:
        raise ValueError("pdqsort cannot sort an array with overlapping elements")

    if values.dtype.isnative and values.flags.c_contiguous and values.flags.aligned:
        working = values
    else:
        working = np.array(values, dtype=native_dtype, order="C", copy=True)

    status = getattr(_LIBRARY, symbol)(int(working.ctypes.data), int(working.size))
    if status != 0:
        raise RuntimeError(f"Mojo pdqsort failed with status {status}")
    if working is not values:
        values[...] = working


def _sort_list(values: list[Any]) -> None:
    if not values:
        return

    if all(isinstance(value, bool) for value in values):
        working = np.asarray(values, dtype=np.uint8)
        _sort_array(working)
        values[:] = (bool(value) for value in working)
        return

    if all(isinstance(value, Integral) and not isinstance(value, bool) for value in values):
        lower = -(1 << 63)
        upper = (1 << 63) - 1
        if any(int(value) < lower or int(value) > upper for value in values):
            raise OverflowError("Python integers must fit in a signed 64-bit value")
        working = np.asarray(values, dtype=np.int64)
        _sort_array(working)
        values[:] = (int(value) for value in working)
        return

    if all(isinstance(value, (float, np.floating)) for value in values):
        working = np.asarray(values, dtype=np.float64)
        _sort_array(working)
        values[:] = (float(value) for value in working)
        return

    raise TypeError("pdqsort lists must contain only homogeneous integers or floats")


def sort(values: Any) -> None:
    """Sort a supported mutable sequence in ascending order, in place."""

    if isinstance(values, np.ndarray):
        _sort_array(values)
        return
    if isinstance(values, list):
        _sort_list(values)
        return
    raise TypeError("pdqsort expects a Python list or a NumPy ndarray")


def pdqsort(values: Any) -> None:
    """Alias for :func:`sort`."""

    sort(values)


class PDQSort:
    """Compatibility API for the original Python ``PDQSort`` class."""

    INSERTION_SORT_THRESHOLD = 24
    NINTHER_THRESHOLD = 128

    @classmethod
    def sort(cls, values: Any) -> None:
        sort(values)
