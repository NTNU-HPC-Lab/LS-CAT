#include "includes.h"
__device__ float activation_function(float x)
{
return 1 / (1 + exp(-x));
}
__global__ void apply_activation_function(float *input, float *output, const int N)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int size = blockDim.x * gridDim.x;

for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
output[idx] = activation_function(input[idx]);
}
}