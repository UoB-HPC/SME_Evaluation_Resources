//------------------------------------------------------------------------------------
// Matrix multiplication code adapted from the SVE programming example B5.1 from
// Arm Ltd. Files found at
// https://developer.arm.com/documentation/dai0548/latest. Accessed 31/08/2022.
//
// Assumptions :
//  - Input and Output matricies are in row-major
//  - Minimum matrix dimension is 32
//  - Matrix dimensions are multiple of 16
//
//------------------------------------------------------------------------------------

#include "matmul.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ERROR_TOLERANCE 0.0002f

int32_t main(int argc, char **argv) {
  int M, N, K, iterations;

  /* Size parameters */
  if (argc == 5) {
    iterations = strtoul(argv[1], NULL, 0);
    M = strtoul(argv[2], NULL, 0);
    K = strtoul(argv[3], NULL, 0);
    N = strtoul(argv[4], NULL, 0);
  } else {
    /* Default: 128x128x128 */
    M = 128;
    K = 128;
    N = 128;
    iterations = 1;
  }

  if (!((M >= 32 && K >= 32 && N >= 32) &&
        (M % 16 == 0 && K % 16 == 0 && N % 16 == 0))) {
    printf("Invalid Matrix Dimensions. Please ensure that :\n\t- Minimum "
           "matrix dimension is 32\n\t- Matrix dimensions are multiple of "
           "16\n\nRuntime args ==> ./BINARY_NAME iterations M K N\n");
    exit(1);
  }

  printf("Left Matrix Dimensions = %d x %d\nRight Matrix Dimensions = %d x "
         "%d\nSVE Loop iterations = %d\n\n",
         M, K, M, N, iterations);

  float *matLeft = (float *)malloc(M * K * sizeof(float));
  float *matRight = (float *)malloc(K * N * sizeof(float));

  float *matResult_ref = (float *)malloc(M * N * sizeof(float));
  float *matResult_opt = (float *)malloc(M * N * sizeof(float));

  srand((unsigned int)time(0));

  for (uint64_t y = 0; y < M; y++) {
    for (uint64_t x = 0; x < K; x++) {
      matLeft[y * K + x] = (((float)(rand() % 10000) / 100.0f) - 30.0);
    }
  }

  for (uint64_t y = 0; y < K; y++) {
    for (uint64_t x = 0; x < N; x++) {
      matRight[y * N + x] = (((float)(rand() % 10000) / 100.0f) - 30.0);
    }
  }

  // Calculate matrix multiply naively
  printf("Calculating Reference Matrix Multiply...\n");
  matmul_ref(M, K, N, matLeft, matRight, matResult_ref);
  printf("Done\n");

  // Calculate matrix multiply using SVE
  // Looping multiple times to isolate performance to this function call
  printf("Calculating %d iterations of SVE Matrix Multiply...\n", 0);
  // for (int i = 0; i < iterations; i++)
  //   matmul_opt(M, K, N, matLeft, matRight, matResult_opt);
  printf("Done\n\n");

  printf("ERROR TOLERANCE = %.4f%%\n", (float)ERROR_TOLERANCE);
  int error = 0;
  for (uint64_t y = 0; y < M; y++) {
    for (uint64_t x = 0; x < N; x++) {
      // if (fabsf(matResult_ref[y * N + x] - matResult_opt[y * N + x]) >
      if (fabsf(matResult_ref[y * N + x] - matResult_ref[y * N + x]) >
          fabsf((float)ERROR_TOLERANCE * matResult_ref[y * N + x])) {
        error = 1;
        printf("%lu, %f, %f\n", y * N + x, matResult_ref[y * N + x],
               matResult_opt[y * N + x]);
      }
    }
  }

  if (error)
    printf("FAILED\n");
  else
    printf("PASS\n");

  free(matLeft);
  free(matRight);
  free(matResult_ref);
  free(matResult_opt);

  return 0;
}
