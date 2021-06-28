#include "includes.h"
__global__ void x_avpb_py_i32 (int* x, int a, int* v, int b, int* y, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
y[idx] += x[idx] * (a * v[idx] + b);
}
}