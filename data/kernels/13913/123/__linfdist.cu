#include "includes.h"
__global__ void __linfdist(float *A, int lda, float *B, int ldb, float *C, int ldc, int d, int nrows, int ncols, float p) {
printf("Warning, Max-abs distance not supported on arch <= 200\n");
}