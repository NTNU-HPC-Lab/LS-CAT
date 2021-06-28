#include "includes.h"
__global__ void __msum(double *A, int lda, double *B, int ldb, double *C, int ldc, int d, int nrows, int ncols, double p) {
printf("Warning, Max-sum multiply not supported on arch <= 200\n");
}