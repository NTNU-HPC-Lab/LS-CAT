#include "includes.h"
__global__ void fill_lower_left_gpu(int *iRow, int *jCol, unsigned int *rind_L, unsigned int *cind_L, const int nnz_L) {
int i = threadIdx.x;

if (i < nnz_L) {
iRow[i] = rind_L[i];
jCol[i] = cind_L[i];
}
}