#include "includes.h"
__device__ float step_function(float v) //Sigmoid function::Activation Function
{
return 1 / (1 + exp(-v));
}
__global__ void apply_step_function(float *input, float *output, const int N)
{
const int pos = blockIdx.x * blockDim.x + threadIdx.x;
const int size = blockDim.x * gridDim.x;

for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
output[idx] = step_function(input[idx]);
}
}