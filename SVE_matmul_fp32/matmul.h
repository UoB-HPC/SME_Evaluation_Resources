//------------------------------------------------------------------------------------
// Matrix multiplication code adapted from the SVE programming example B5.1 from
// Arm Ltd. Files found at
// https://developer.arm.com/documentation/dai0548/latest. Accessed 31/08/2022.
//------------------------------------------------------------------------------------

#include <stdint.h>

void matmul_ref(const uint64_t rows_l, const uint64_t cols_l,
                const uint64_t cols_r, const float *restrict input_left,
                const float *restrict input_right, float *restrict output);

void matmul_opt(const uint64_t M, const uint64_t K, const uint64_t N,
                float *restrict input_left, float *restrict input_right,
                float *restrict output);