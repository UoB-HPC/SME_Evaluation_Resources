//------------------------------------------------------------------------------------
// Matrix multiplication code adapted from an SME programming example provided
// by Arm Ltd.
//------------------------------------------------------------------------------------

#include <stdint.h>

void matmul_ref(const uint64_t rows_l, const uint64_t cols_l,
                const uint64_t cols_r, const float *restrict input_left,
                const float *restrict input_right, float *restrict output);

void matmul_opt(const uint64_t rows_l, const uint64_t cols_l,
                const uint64_t cols_r, const float *restrict input_left,
                const float *restrict input_right, float *restrict output);

void matmul_opt_test(const uint64_t rows_l, const uint64_t cols_l,
                     const uint64_t cols_r, const float *restrict input_left,
                     const float *restrict input_right, float *restrict output);

void preprocess_l(const uint64_t rows, const uint64_t cols,
                  const float *restrict a, float *restrict a_mod);

inline uint64_t sve_cntw() {
  uint64_t cnt;
  asm volatile(".4byte 0xd503437f\n" // smstart sm
               "cntw %[res]\n"
               ".4byte 0xd503427f \n" // smstop sm
               : [res] "=r"(cnt)
               :
               : "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9",
                 "p10", "p11", "p12", "p13", "p14", "p15", "z0", "z1", "z2",
                 "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12",
                 "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21",
                 "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30",
                 "z31");
  return cnt;
}