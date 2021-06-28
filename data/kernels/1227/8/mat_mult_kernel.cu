#include "includes.h"
__global__ void mat_mult_kernel(int *mat_a, int *mat_b, int *result, int a_rows, int a_cols, int b_cols) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
while (tid < a_rows) {
for (int j = 0; j < b_cols; j++) {
int temp_res = 0;
for (int k = 0; k < a_cols; k++) {
temp_res  += mat_a[tid * a_cols + k] * mat_b[k * b_cols + j];
}
result[tid * b_cols + j] = temp_res;
}
tid += blockDim.x * gridDim.x;
}
}