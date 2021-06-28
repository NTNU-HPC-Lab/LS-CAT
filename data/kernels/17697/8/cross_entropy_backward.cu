#include "includes.h"
extern "C" {
}
__global__ void cross_entropy_backward(const float* x, float* dx, const float* t, float* dy, unsigned int len) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < len) {
dx[tid] = dy[0] * (x[tid] - t[tid]);
}
}