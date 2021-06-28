#include "includes.h"
__global__ void bcnn_op_cuda_relu_kernel(int n, float *x, float *y) {
int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
if (i < n) {
y[i] = x[i] * (x[i] > 0);
}
return;
}