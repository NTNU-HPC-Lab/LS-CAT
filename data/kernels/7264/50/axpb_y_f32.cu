#include "includes.h"
__global__ void axpb_y_f32 (float a, float* x, float b, float* y, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
y[idx] *= a * x[idx] + b;
}
}