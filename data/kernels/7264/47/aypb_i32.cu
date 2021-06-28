#include "includes.h"
__global__ void aypb_i32 (int a, int* y, int b, int len) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) {
y[idx] = a * y[idx] + b;
}
}