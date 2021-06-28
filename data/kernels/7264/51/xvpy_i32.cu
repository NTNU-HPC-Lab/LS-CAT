#include "includes.h"
__global__ void xvpy_i32 (int* x, int* v, int* y, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
y[idx] += x[idx] * v[idx];
}
}