#include "includes.h"
__global__ void apply_grad(float *output, float *grad, const int N)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int size = blockDim.x * gridDim.x;

for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
output[idx] += dt * grad[idx];
}
}