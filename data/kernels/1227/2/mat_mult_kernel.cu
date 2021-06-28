#include "includes.h"
__global__ void mat_mult_kernel(int *a, int *b, int *c, int mat_rows, int mat_cols) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < mat_rows) {
int res = 0;
for (int i = 0; i < mat_cols; i++) {
res += a[tid * mat_cols + i] * b[i];
}
c[tid] = res;
tid += blockDim.x * gridDim.x;
}
}