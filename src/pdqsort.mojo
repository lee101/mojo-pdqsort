from std.memory import stack_allocation


comptime INSERTION_SORT_THRESHOLD = 24
comptime NINTHER_THRESHOLD = 128
comptime PARTIAL_INSERTION_LIMIT = 8
comptime PARTITION_BLOCK_SIZE = 64
comptime PARALLEL_SORT_THRESHOLD = 262144


def _less[dtype: DType](left: Scalar[dtype], right: Scalar[dtype]) -> Bool:
    return left < right or (left == left and right != right)


def _swap[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], i: Int, j: Int
):
    var value = data[i]
    data[i] = data[j]
    data[j] = value


def _median_of_three[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], a: Int, b: Int, c: Int
) -> Int:
    if _less[dtype](data[a], data[b]):
        if _less[dtype](data[b], data[c]):
            return b
        if _less[dtype](data[a], data[c]):
            return c
        return a
    if _less[dtype](data[a], data[c]):
        return a
    if _less[dtype](data[b], data[c]):
        return c
    return b


def _choose_pivot[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], begin: Int, end: Int
) -> Int:
    var size = end - begin
    var middle = begin + size // 2
    if size <= NINTHER_THRESHOLD:
        return _median_of_three[dtype](data, begin, middle, end - 1)

    var step = size // 8
    var first = _median_of_three[dtype](data, begin, begin + step, begin + 2 * step)
    var center = _median_of_three[dtype](data, middle - step, middle, middle + step)
    var last = _median_of_three[dtype](data, end - 1 - 2 * step, end - 1 - step, end - 1)
    return _median_of_three[dtype](data, first, center, last)


def _insertion_sort[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], begin: Int, end: Int
):
    var i = begin + 1
    while i < end:
        var value = data[i]
        var hole = i
        while hole > begin and _less[dtype](value, data[hole - 1]):
            data[hole] = data[hole - 1]
            hole -= 1
        data[hole] = value
        i += 1


def _partial_insertion_sort[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], begin: Int, end: Int
) -> Bool:
    if end - begin < 2:
        return True

    var moves = 0
    var i = begin + 1
    while i < end:
        if _less[dtype](data[i], data[i - 1]):
            var value = data[i]
            var hole = i
            while hole > begin and _less[dtype](value, data[hole - 1]):
                data[hole] = data[hole - 1]
                hole -= 1
                moves += 1
            data[hole] = value
            if moves > PARTIAL_INSERTION_LIMIT:
                return False
        i += 1
    return True


def _is_sorted[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], begin: Int, end: Int
) -> Bool:
    var i = begin
    while i + 1 < end:
        if _less[dtype](data[i + 1], data[i]):
            return False
        i += 1
    return True


def _sift_down[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], begin: Int, root: Int, count: Int
):
    var current = root
    while 2 * current + 1 < count:
        var child = 2 * current + 1
        if child + 1 < count and _less[dtype](data[begin + child], data[begin + child + 1]):
            child += 1
        if not _less[dtype](data[begin + current], data[begin + child]):
            return
        _swap[dtype](data, begin + current, begin + child)
        current = child


def _heap_sort[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], begin: Int, end: Int
):
    var count = end - begin
    var parent = count // 2
    while parent > 0:
        parent -= 1
        _sift_down[dtype](data, begin, parent, count)

    var remaining = count
    while remaining > 1:
        remaining -= 1
        _swap[dtype](data, begin, begin + remaining)
        _sift_down[dtype](data, begin, 0, remaining)


def _partition_right[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    begin: Int,
    end: Int,
    pivot_index: Int,
) -> Tuple[Int, Bool]:
    _swap[dtype](data, begin, pivot_index)
    var pivot = data[begin]
    var first = begin + 1
    var last = end

    while first < last and _less[dtype](data[first], pivot):
        first += 1
    while first < last and not _less[dtype](data[last - 1], pivot):
        last -= 1

    var already_partitioned = first >= last
    var left_offsets = stack_allocation[PARTITION_BLOCK_SIZE, Int]()
    var right_offsets = stack_allocation[PARTITION_BLOCK_SIZE, Int]()
    var left_count = 0
    var right_count = 0
    var left_start = 0
    var right_start = 0

    while last - first > 2 * PARTITION_BLOCK_SIZE:
        if left_count == 0:
            left_start = 0
            var i = 0
            while i < PARTITION_BLOCK_SIZE:
                left_offsets[left_count] = i
                left_count += Int(not _less[dtype](data[first + i], pivot))
                i += 1

        if right_count == 0:
            right_start = 0
            var i = 0
            while i < PARTITION_BLOCK_SIZE:
                right_offsets[right_count] = i + 1
                right_count += Int(_less[dtype](data[last - i - 1], pivot))
                i += 1

        var count = min(left_count, right_count)
        var i = 0
        while i < count:
            _swap[dtype](
                data,
                first + left_offsets[left_start + i],
                last - right_offsets[right_start + i],
            )
            i += 1
        if count > 0:
            already_partitioned = False
        left_count -= count
        right_count -= count
        left_start += count
        right_start += count
        if left_count == 0:
            first += PARTITION_BLOCK_SIZE
        if right_count == 0:
            last -= PARTITION_BLOCK_SIZE

    while True:
        while first < last and _less[dtype](data[first], pivot):
            first += 1
        while first < last and not _less[dtype](data[last - 1], pivot):
            last -= 1
        if first >= last:
            break
        last -= 1
        _swap[dtype](data, first, last)
        already_partitioned = False
        first += 1

    var pivot_position = first - 1
    _swap[dtype](data, begin, pivot_position)
    return (pivot_position, already_partitioned)


def _partition_left[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    begin: Int,
    end: Int,
    pivot_index: Int,
) -> Int:
    _swap[dtype](data, begin, pivot_index)
    var pivot = data[begin]
    var first = begin + 1
    var last = end - 1

    while first <= last and not _less[dtype](pivot, data[first]):
        first += 1
    while first <= last and _less[dtype](pivot, data[last]):
        last -= 1

    while first <= last:
        _swap[dtype](data, first, last)
        first += 1
        last -= 1
        while first <= last and not _less[dtype](pivot, data[first]):
            first += 1
        while first <= last and _less[dtype](pivot, data[last]):
            last -= 1

    return first


def _break_patterns[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    begin: Int,
    pivot: Int,
    end: Int,
):
    var left_size = pivot - begin
    if left_size >= INSERTION_SORT_THRESHOLD:
        var offset = left_size // 4
        _swap[dtype](data, begin, begin + offset)
        _swap[dtype](data, pivot - 1, pivot - 1 - offset)
        if left_size > NINTHER_THRESHOLD:
            _swap[dtype](data, begin + 1, begin + offset + 1)
            _swap[dtype](data, begin + 2, begin + offset + 2)

    var right_begin = pivot + 1
    var right_size = end - right_begin
    if right_size >= INSERTION_SORT_THRESHOLD:
        var offset = right_size // 4
        _swap[dtype](data, right_begin, right_begin + offset)
        _swap[dtype](data, end - 1, end - 1 - offset)
        if right_size > NINTHER_THRESHOLD:
            _swap[dtype](data, right_begin + 1, right_begin + offset + 1)
            _swap[dtype](data, right_begin + 2, right_begin + offset + 2)


def _pdqsort_loop[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]],
    begin: Int,
    end: Int,
    bad_allowed: Int,
    leftmost: Bool,
):
    var range_begin = begin
    var bad = bad_allowed
    var is_leftmost = leftmost

    while True:
        var size = end - range_begin
        if size < 2:
            return
        if size <= INSERTION_SORT_THRESHOLD:
            _insertion_sort[dtype](data, range_begin, end)
            return

        var pivot_choice = _choose_pivot[dtype](data, range_begin, end)

        if not is_leftmost and not _less[dtype](data[range_begin - 1], data[pivot_choice]):
            range_begin = _partition_left[dtype](data, range_begin, end, pivot_choice)
            is_leftmost = False
            continue

        var partition = _partition_right[dtype](data, range_begin, end, pivot_choice)
        var pivot = partition[0]
        var already_partitioned = partition[1]
        var left_size = pivot - range_begin
        var right_size = end - pivot - 1
        var unbalanced = left_size < size // 8 or right_size < size // 8

        if unbalanced:
            bad -= 1
            if bad == 0:
                _heap_sort[dtype](data, range_begin, end)
                return
            _break_patterns[dtype](data, range_begin, pivot, end)
        elif already_partitioned:
            if _partial_insertion_sort[dtype](data, range_begin, pivot) and _partial_insertion_sort[dtype](data, pivot + 1, end):
                return

        _pdqsort_loop[dtype](data, range_begin, pivot, bad, is_leftmost)
        range_begin = pivot + 1
        is_leftmost = False


def _bad_partition_limit(length: Int) -> Int:
    var limit = 0
    var n = length
    while n > 1:
        limit += 1
        n //= 2
    return limit


def _sort[dtype: DType](
    data: UnsafePointer[Scalar[dtype], AnyOrigin[mut=True]], length: Int
):
    if length < 2:
        return
    if length >= INSERTION_SORT_THRESHOLD and _is_sorted[dtype](data, 0, length):
        return

    var bad_allowed = _bad_partition_limit(length)
    if length < PARALLEL_SORT_THRESHOLD:
        _pdqsort_loop[dtype](data, 0, length, bad_allowed, True)
        return

    var pivot_choice = _choose_pivot[dtype](data, 0, length)
    var partition = _partition_right[dtype](data, 0, length, pivot_choice)
    var pivot = partition[0]
    var left_size = pivot
    var right_size = length - pivot - 1

    if left_size >= length // 8 and right_size >= length // 8:
        var subpivots = stack_allocation[2, Int]()

        @parameter
        def split_half(index: Int):
            if index == 0:
                var choice = _choose_pivot[dtype](data, 0, pivot)
                subpivots[0] = _partition_right[dtype](
                    data, 0, pivot, choice
                )[0]
            else:
                var choice = _choose_pivot[dtype](data, pivot + 1, length)
                subpivots[1] = _partition_right[dtype](
                    data, pivot + 1, length, choice
                )[0]

        split_half(0)
        split_half(1)
        var left_pivot = subpivots[0]
        var right_pivot = subpivots[1]

        @parameter
        def sort_quarter(index: Int):
            if index == 0:
                _pdqsort_loop[dtype](
                    data, 0, left_pivot, _bad_partition_limit(left_pivot), True
                )
            elif index == 1:
                _pdqsort_loop[dtype](
                    data,
                    left_pivot + 1,
                    pivot,
                    _bad_partition_limit(pivot - left_pivot - 1),
                    True,
                )
            elif index == 2:
                _pdqsort_loop[dtype](
                    data,
                    pivot + 1,
                    right_pivot,
                    _bad_partition_limit(right_pivot - pivot - 1),
                    True,
                )
            else:
                _pdqsort_loop[dtype](
                    data,
                    right_pivot + 1,
                    length,
                    _bad_partition_limit(length - right_pivot - 1),
                    True,
                )

        sort_quarter(0)
        sort_quarter(1)
        sort_quarter(2)
        sort_quarter(3)
    else:
        _pdqsort_loop[dtype](data, 0, pivot, bad_allowed, True)
        _pdqsort_loop[dtype](data, pivot + 1, length, bad_allowed, True)


@export("pdqsort_i8")
def pdqsort_i8(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[Int8, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.int8](data, length)
    return 0


@export("pdqsort_i16")
def pdqsort_i16(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[Int16, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.int16](data, length)
    return 0


@export("pdqsort_i32")
def pdqsort_i32(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[Int32, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.int32](data, length)
    return 0


@export("pdqsort_i64")
def pdqsort_i64(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.int64](data, length)
    return 0


@export("pdqsort_u8")
def pdqsort_u8(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[UInt8, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.uint8](data, length)
    return 0


@export("pdqsort_u16")
def pdqsort_u16(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[UInt16, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.uint16](data, length)
    return 0


@export("pdqsort_u32")
def pdqsort_u32(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.uint32](data, length)
    return 0


@export("pdqsort_u64")
def pdqsort_u64(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[UInt64, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.uint64](data, length)
    return 0


@export("pdqsort_f32")
def pdqsort_f32(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[Float32, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.float32](data, length)
    return 0


@export("pdqsort_f64")
def pdqsort_f64(address: Int, length: Int) abi("C") -> Int:
    if length < 0 or (length > 1 and address == 0):
        return -1
    if length > 1:
        var data = UnsafePointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=address)
        _sort[DType.float64](data, length)
    return 0
