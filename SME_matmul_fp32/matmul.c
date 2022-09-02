//------------------------------------------------------------------------------------
// Matrix multiplication code adapted from an SME programming example provided
// by Arm Ltd.
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

void matmul_opt(const uint64_t rows_l, const uint64_t cols_l,
                const uint64_t cols_r, const float *restrict input_left,
                const float *restrict input_right, float *restrict output) {
  asm volatile(
      "stp	x19, x20, [sp, #-48]!\n"
      "stp	x21, x22, [sp, #16]\n"
      "stp	x23, x24, [sp, #32]\n"
      ".4byte 0xd503477f\n" // smstart
      "cntw	x6\n"
      "mul	x22, x6, x1\n"
      "mul	x23, x6, x2\n"
      "add	x18, x23, x2\n"
      "add	x11, x4, x2, lsl #2\n"
      "mov	x12, #0\n"
      "add	x19, x2, x6\n"
      "add	x20, x22, x6\n"
      "whilelt	p2.s, x12, x0\n"
      "sub	w6, w6, #2\n"
      "incw	x12\n"
      "whilelt	p3.s, x12, x0\n"
      "mov	x16, x4\n"
      "mov	x9, x5\n"
      "whilelt	p0.b, x16, x11\n"
      "mov	x7, x3\n"
      "mov	x17, x16\n"
      "mov	x10, x9\n"
      "addvl	x21, x9, #1\n"
      "addvl	x16, x16, #1\n"
      "whilelt	p1.b, x16, x11\n"
      "add	x8, x3, x22, lsl #2\n"
      "addvl	x15, x8, #-1\n"
      "ld1w	{ z1.s }, p2/z, [x7]\n"
      ".4byte 0xc00800ff\n" // zero	{za}
      "ld1w	{ z2.s }, p0/z, [x17]\n"
      ".4byte 0x80820820\n" // fmopa	za0.s, p2/m, p0/m, z1.s, z2.s
      "ld1w	{ z5.s }, p3/z, [x7, x22, lsl #2]\n"
      "addvl	x7, x7, #1\n"
      ".4byte 0x80820ca2\n" // fmopa	za2.s, p3/m, p0/m, z5.s, z2.s
      "ld1w	{ z3.s }, p1/z, [x17, #1, mul vl]\n"
      ".4byte 0x80832821\n" // fmopa	za1.s, p2/m, p1/m, z1.s, z3.s
      "ld1w	{ z0.s }, p2/z, [x7]\n"
      ".4byte 0x80832ca3\n" // fmopa	za3.s, p3/m, p1/m, z5.s, z3.s
      "ld1w	{ z6.s }, p0/z, [x17, x2, lsl #2]\n"
      ".4byte 0x80860800\n" // fmopa	za0.s, p2/m, p0/m, z0.s, z6.s
      "ld1w	{ z4.s }, p3/z, [x7, x22, lsl #2]\n"
      ".4byte 0x80860c82\n" // fmopa	za2.s, p3/m, p0/m, z4.s, z6.s
      "ld1w	{ z7.s }, p1/z, [x17, x19, lsl #2]\n"
      "add	x17, x17, x2, lsl #3\n"
      ".4byte 0x80872801\n" // fmopa	za1.s, p2/m, p1/m, z0.s, z7.s
      "ld1w	{ z1.s }, p2/z, [x7, #1, mul vl]\n"
      ".4byte 0x80872c83\n" // fmopa	za3.s, p3/m, p1/m, z4.s, z7.s
      "ld1w	{ z2.s }, p0/z, [x17]\n"
      ".4byte 0x80820820\n" // fmopa	za0.s, p2/m, p0/m, z1.s, z2.s
      "ld1w	{ z5.s }, p3/z, [x7, x20, lsl #2]\n"
      "addvl	x7, x7, #2\n"
      "cmp	x7, x15\n"
      ".4byte 0x54fffda4\n" // b.mi	0x8000014c <matmul_opt+0x84>
      ".4byte 0x80820ca2\n" // fmopa	za2.s, p3/m, p0/m, z5.s, z2.s
      "ld1w	{ z3.s }, p1/z, [x17, #1, mul vl]\n"
      ".4byte 0x80832821\n" // fmopa	za1.s, p2/m, p1/m, z1.s, z3.s
      ".4byte 0x80832ca3\n" // fmopa	za3.s, p3/m, p1/m, z5.s, z3.s
      "add	x17, x17, x2, lsl #2\n"
      "cmp	x7, x8\n"
      ".4byte 0x54000125\n" // b.pl	0x800001d8 <.Ktail_end>
      // <.Ktail_start>:
      "ld1w	{ z1.s }, p2/z, [x7]\n"
      "ld1w	{ z2.s }, p0/z, [x17]\n"
      "ld1w	{ z3.s }, p1/z, [x17, #1, mul vl]\n"
      ".4byte 0x80820820\n" // fmopa	za0.s, p2/m, p0/m, z1.s, z2.s
      "ld1w	{ z5.s }, p3/z, [x7, x22, lsl #2]\n"
      ".4byte 0x80832821\n" // fmopa	za1.s, p2/m, p1/m, z1.s, z3.s
      ".4byte 0x80820ca2\n" // fmopa	za2.s, p3/m, p0/m, z5.s, z2.s
      ".4byte 0x80832ca3\n" // fmopa	za3.s, p3/m, p1/m, z5.s, z3.s
      // <.Ktail_end>:
      "mov	w13, #0\n"
      ".4byte 0x25314044\n" // psel	p4, p0, p2.s[w13, 0]
      ".4byte 0x25314445\n" // psel	p5, p1, p2.s[w13, 0]
      ".4byte 0x25314066\n" // psel	p6, p0, p3.s[w13, 0]
      ".4byte 0x25314467\n" // psel	p7, p1, p3.s[w13, 0]
      ".4byte 0xe0bf3140\n" // st1w	{za0h.s[w13, 0]}, p4, [x10]
      ".4byte 0xe0bf36a4\n" // st1w	{za1h.s[w13, 0]}, p5, [x21]
      ".4byte 0xe0b73948\n" // st1w	{za2h.s[w13, 0]}, p6, [x10, x23, lsl #2]
      ".4byte 0xe0b73eac\n" // st1w	{za3h.s[w13, 0]}, p7, [x21, x23, lsl #2]
      ".4byte 0x25714044\n" // psel	p4, p0, p2.s[w13, 1]
      ".4byte 0x25714445\n" // psel	p5, p1, p2.s[w13, 1]
      ".4byte 0x25714066\n" // psel	p6, p0, p3.s[w13, 1]
      ".4byte 0x25714467\n" // psel	p7, p1, p3.s[w13, 1]
      ".4byte 0xe0a23141\n" // st1w	{za0h.s[w13, 1]}, p4, [x10, x2, lsl #2]
      ".4byte 0xe0a236a5\n" // st1w	{za1h.s[w13, 1]}, p5, [x21, x2, lsl #2]
      ".4byte 0xe0b23949\n" // st1w	{za2h.s[w13, 1]}, p6, [x10, x18, lsl #2]
      ".4byte 0xe0b23ead\n" // st1w	{za3h.s[w13, 1]}, p7, [x21, x18, lsl #2]
      "add	x10, x10, x2, lsl #3\n"
      "add	x21, x21, x2, lsl #3\n"
      "add	w13, w13, #2\n"
      ".4byte 0x25314044\n" // psel	p4, p0, p2.s[w13, 0]
      ".4byte 0x25314445\n" // psel	p5, p1, p2.s[w13, 0]
      ".4byte 0x25314066\n" // psel	p6, p0, p3.s[w13, 0]
      ".4byte 0x25314467\n" // psel	p7, p1, p3.s[w13, 0]
      ".4byte 0xe0bf3140\n" // st1w	{za0h.s[w13, 0]}, p4, [x10]
      ".4byte 0xe0bf36a4\n" // st1w	{za1h.s[w13, 0]}, p5, [x21]
      ".4byte 0xe0b73948\n" // st1w	{za2h.s[w13, 0]}, p6, [x10, x23, lsl #2]
      ".4byte 0xe0b73eac\n" // st1w	{za3h.s[w13, 0]}, p7, [x21, x23, lsl #2]
      "cmp	w13, w6\n"
      ".4byte 0x54fffd84\n" // b.mi	0x800001fc <.Ktail_end+0x24>
      ".4byte 0x25714044\n" // psel	p4, p0, p2.s[w13, 1]
      ".4byte 0x25714445\n" // psel	p5, p1, p2.s[w13, 1]
      ".4byte 0x25714066\n" // psel	p6, p0, p3.s[w13, 1]
      ".4byte 0x25714467\n" // psel	p7, p1, p3.s[w13, 1]
      ".4byte 0xe0a23141\n" // st1w	{za0h.s[w13, 1]}, p4, [x10, x2, lsl #2]
      ".4byte 0xe0a236a5\n" // st1w	{za1h.s[w13, 1]}, p5, [x21, x2, lsl #2]
      ".4byte 0xe0b23949\n" // st1w	{za2h.s[w13, 1]}, p6, [x10, x18, lsl #2]
      ".4byte 0xe0b23ead\n" // st1w	{za3h.s[w13, 1]}, p7, [x21, x18, lsl #2]
      "addvl	x9, x9, #2\n"
      "addvl	x16, x16, #1\n"
      "whilelt	p0.b, x16, x11\n"
      ".4byte 0x54fff4c4\n" // b.mi	0x80000114 <matmul_opt+0x4c>
      "add	x3, x3, x22, lsl #3\n"
      "add	x5, x5, x23, lsl #3\n"
      "incw	x12\n"
      "whilelt	p2.s, x12, x0\n"
      ".4byte 0x54fff384\n" // b.mi	0x80000100 <matmul_opt+0x38>
      ".4byte 0xd503467f\n" // smstop
      "ldp	x23, x24, [sp, #32]\n"
      "ldp	x21, x22, [sp, #16]\n"
      "ldp	x19, x20, [sp], #48\n"
      "ret\n");
}

void preprocess_l(const uint64_t rows, const uint64_t cols,
                  const float *restrict a, float *restrict a_mod) {
  asm volatile(
      ".4byte 0xd503477f\n" // smstart
      "cntw	x4\n"
      "mul	x11, x4, x1\n"
      "lsl	x14, x1, #1\n"
      "add	x15, x14, x1\n"
      "cnth	x13\n"
      "add	x10, x4, x4, lsl #1\n"
      "mov	x7, #0\n"
      "whilelt	p0.s, x7, x0\n"
      "ptrue	p9.s\n"
      "mov	x8, x2\n"
      "mov	x9, x3\n"
      "add	x5, x2, x1, lsl #2\n"
      "whilelt	p8.b, x8, x5\n"
      "mov	x6, x8\n"
      "mov	w12, #0\n"
      ".4byte 0x25306002\n" // psel	p2, p8, p0.s[w12, 0]
      ".4byte 0x25706003\n" // psel	p3, p8, p0.s[w12, 1]
      ".4byte 0x25b06004\n" // psel	p4, p8, p0.s[w12, 2]
      ".4byte 0x25f06005\n" // psel	p5, p8, p0.s[w12, 3]
      ".4byte 0xe09f08c0\n" // ld1w	{za0h.s[w12, 0]}, p2/z, [x6]
      ".4byte 0xe0810cc1\n" // ld1w	{za0h.s[w12, 1]}, p3/z, [x6, x1, lsl #2]
      ".4byte 0xe08e10c2\n" // ld1w	{za0h.s[w12, 2]}, p4/z, [x6, x14, lsl
                            // #2]
      ".4byte 0xe08f14c3\n" // ld1w	{za0h.s[w12, 3]}, p5/z, [x6, x15, lsl
                            // #2]
      "add	x6, x6, x1, lsl #4\n"
      "add	w12, w12, #4\n"
      "cmp	w12, w4\n"
      ".4byte 0x54fffea4\n" // b.mi	0x800002e8 <preprocess_l+0x40>
      "mov	w12, #0\n"
      // <.Store_loop>:
      ".4byte 0x25306502\n" // psel	p2, p9, p8.s[w12, 0]
      ".4byte 0x25706503\n" // psel	p3, p9, p8.s[w12, 1]
      ".4byte 0x25b06504\n" // psel	p4, p9, p8.s[w12, 2]
      ".4byte 0x25f06505\n" // psel	p5, p9, p8.s[w12, 3]
      ".4byte 0xe0bf8920\n" // st1w	{za0v.s[w12, 0]}, p2, [x9]
      ".4byte 0xe0a48d21\n" // st1w	{za0v.s[w12, 1]}, p3, [x9, x4, lsl #2]
      ".4byte 0xe0ad9122\n" // st1w	{za0v.s[w12, 2]}, p4, [x9, x13, lsl #2]
      ".4byte 0xe0aa9523\n" // st1w	{za0v.s[w12, 3]}, p5, [x9, x10, lsl #2]
      "addvl	x9, x9, #4\n"
      "add	w12, w12, #4\n"
      "cmp	w12, w4\n"
      ".4byte 0x54fffea4\n" // b.mi	0x8000031c <.Store_loop>
      "addvl	x8, x8, #1\n"
      "whilelt	p8.b, x8, x5\n"
      ".4byte 0x54fffc64\n" // b.mi	0x800002e0 <preprocess_l+0x38>
      "add	x3, x3, x11, lsl #2\n"
      "add	x2, x2, x11, lsl #2\n"
      "incw	x7\n"
      "whilelt	p0.s, x7, x0\n"
      ".4byte 0x54fffb44\n" // b.mi	0x800002d0 <preprocess_l+0x28>
      ".4byte 0xd503467f\n" // smstop
      "ret\n");
}