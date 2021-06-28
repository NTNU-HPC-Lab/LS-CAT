#include "includes.h"
__global__ void x_avpb_py_f32 (float* x, float a, float* v, float b, float* y, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
y[idx] += x[idx] * (a * v[idx] + b);
}
}