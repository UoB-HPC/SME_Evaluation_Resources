//------------------------------------------------------------------------------------
// Matrix multiplication code adapted from the SVE programming example B5.1 from
// Arm Ltd. Files found at
// https://developer.arm.com/documentation/dai0548/latest. Accessed 31/08/2022.
//------------------------------------------------------------------------------------

#include "matmul.h"
#include <arm_sve.h>

void matmul_ref(const uint64_t rows_l, const uint64_t cols_l,
                const uint64_t cols_r, const float *restrict input_left,
                const float *restrict input_right, float *restrict output) {
  for (uint64_t x = 0; x < rows_l; ++x) {
    for (uint64_t y = 0; y < cols_r; ++y) {
      float acc = 0.0f;

      for (uint64_t z = 0; z < cols_l; ++z) {
        acc += input_left[(x * cols_l) + z] * input_right[(z * cols_r) + y];
      }

      output[(x * cols_r) + y] = acc;
    }
  }
}

void matmul_opt(const uint64_t M, const uint64_t K, const uint64_t N,
                float *restrict input_left, float *restrict input_right,
                float *restrict output) {
  svbool_t p32_all = svptrue_b32();
  uint64_t vl = svcntw();
  uint64_t offset_in_1, offset_in_2, offset_in_3;
  uint64_t offset_out_1, offset_out_2, offset_out_3;

  float32_t *ptr_in_left, *ptr_in_right, *ptr_out;

  svfloat32_t acc0, acc1, acc2, acc3;
  svfloat32_t in_right_0, in_right_1;
  svfloat32_t in_left_0, in_left_1, in_left_2, in_left_3;

  offset_in_1 = K;
  offset_in_2 = 2 * K;
  offset_in_3 = 3 * K;

  offset_out_1 = N;
  offset_out_2 = 2 * N;
  offset_out_3 = 3 * N;

  for (uint64_t x = 0; x < M; x += 4) {
    ptr_out = &output[x * N];

    for (uint64_t y = 0; y < N; y += vl) {
      acc0 = svdup_f32(0.0);
      acc1 = svdup_f32(0.0);
      acc2 = svdup_f32(0.0);
      acc3 = svdup_f32(0.0);

      ptr_in_left = &input_left[x * K];
      ptr_in_right = &input_right[y];

      for (uint64_t z = 0; z < K; z += 2) {
        in_right_0 = svld1(p32_all, ptr_in_right);
        in_right_1 = svld1(p32_all, &ptr_in_right[offset_out_1]);

        in_left_0 = svld1rq(p32_all, ptr_in_left);
        in_left_1 = svld1rq(p32_all, &ptr_in_left[offset_in_1]);
        in_left_2 = svld1rq(p32_all, &ptr_in_left[offset_in_2]);
        in_left_3 = svld1rq(p32_all, &ptr_in_left[offset_in_3]);

        acc0 = svmla_lane(acc0, in_right_0, in_left_0, 0);
        acc0 = svmla_lane(acc0, in_right_1, in_left_0, 1);

        acc1 = svmla_lane(acc1, in_right_0, in_left_1, 0);
        acc1 = svmla_lane(acc1, in_right_1, in_left_1, 1);

        acc2 = svmla_lane(acc2, in_right_0, in_left_2, 0);
        acc2 = svmla_lane(acc2, in_right_1, in_left_2, 1);

        acc3 = svmla_lane(acc3, in_right_0, in_left_3, 0);
        acc3 = svmla_lane(acc3, in_right_1, in_left_3, 1);

        ptr_in_right += 2 * N;
        ptr_in_left += 2;
      }

      svst1(p32_all, ptr_out, acc0);
      svst1(p32_all, &ptr_out[offset_out_1], acc1);
      svst1(p32_all, &ptr_out[offset_out_2], acc2);
      svst1(p32_all, &ptr_out[offset_out_3], acc3);

      ptr_out += vl;
    }
  }
}
