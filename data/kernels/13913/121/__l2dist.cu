#include "includes.h"
__global__ void __l2dist(float *A, int lda, float *B, int ldb, float *C, int ldc, int d, int nrows, int ncols, float p) {
printf("Warning, L2dist not supported on arch <= 200\n");
}