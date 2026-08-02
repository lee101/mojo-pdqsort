#ifndef MOJO_PDQSort_H
#define MOJO_PDQSort_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

intptr_t pdqsort_i8(intptr_t address, intptr_t length);
intptr_t pdqsort_i16(intptr_t address, intptr_t length);
intptr_t pdqsort_i32(intptr_t address, intptr_t length);
intptr_t pdqsort_i64(intptr_t address, intptr_t length);
intptr_t pdqsort_u8(intptr_t address, intptr_t length);
intptr_t pdqsort_u16(intptr_t address, intptr_t length);
intptr_t pdqsort_u32(intptr_t address, intptr_t length);
intptr_t pdqsort_u64(intptr_t address, intptr_t length);
intptr_t pdqsort_f32(intptr_t address, intptr_t length);
intptr_t pdqsort_f64(intptr_t address, intptr_t length);

#ifdef __cplusplus
}
#endif

#endif
