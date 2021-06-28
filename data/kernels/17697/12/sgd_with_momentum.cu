#include "includes.h"
extern "C" {
}
__global__ void sgd_with_momentum(float* w, const float* dw, float learning_rate, float momentum, float* v, unsigned int len) {
int tid = blockIdx.x*blockDim.x + threadIdx.x;
if (tid < len) {
v[tid] = momentum * v[tid] + dw[tid];
w[tid] -= learning_rate * v[tid];
}
}