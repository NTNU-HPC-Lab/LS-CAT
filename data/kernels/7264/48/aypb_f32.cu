#include "includes.h"
__global__ void aypb_f32 (float a, float* y, float b, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
y[idx] = a * y[idx] + b;
}
}