#include "includes.h"
extern "C" {
}
__global__ void broadcast_backward(float* dx, const float* dy, unsigned int c, unsigned int len) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < len) {
atomicAdd(&dx[tid % c], dy[tid]);
}
}