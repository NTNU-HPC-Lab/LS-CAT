#include "includes.h"
extern "C" {
}
__global__ void broadcast(const float* x, float* y, unsigned int c, unsigned int len) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < len) {
y[tid] = x[tid % c];
}
}