#include "includes.h"
__global__ void axpb_y_i32 (int a, int* x, int b, int* y, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
y[idx] *= a * x[idx] + b;
}
}