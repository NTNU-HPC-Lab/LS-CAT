#include "includes.h"
__global__ void nllLoss(float *x, int x_stride, float *y, int* target) {
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int offset = tid * x_stride + target[tid];
y[tid] = -1 * x[offset];
}