#include "includes.h"

__global__ void encode_cols_kernel(float *a, uint32_t* b, int m, int n) {
int j = blockIdx.x * blockDim.x + threadIdx.x;
int i = blockIdx.y * blockDim.y + threadIdx.y;
int i32 = i*ENCODE_BITS;
if (j < n && i32 < m) {
uint32_t r = 0;
for(int k = 0; j + n * (i32 + k)< m * n && k < ENCODE_BITS; k++){
r |= (a[j + n * (i32 + k)]>0)<<k;
}
b[j + n * i] = r;
}
}