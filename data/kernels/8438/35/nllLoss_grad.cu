#include "includes.h"
__global__ void nllLoss_grad(int x_stride, float *yGrad, int* target, float* xGrad) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int offset = tid * x_stride + target[tid];
xGrad[offset] += -1 * yGrad[tid];
}