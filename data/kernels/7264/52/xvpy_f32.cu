#include "includes.h"
__global__ void xvpy_f32 (float* x, float* v, float* y, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
y[idx] += x[idx] * v[idx];
}
}