#include "includes.h"
__global__ void __l2dist(double *A, int lda, double *B, int ldb, double *C, int ldc, int d, int nrows, int ncols, double p) {
printf("Warning, L2dist not supported on arch <= 200\n");
}