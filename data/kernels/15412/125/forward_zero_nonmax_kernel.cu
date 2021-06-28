#include "includes.h"
__global__ void forward_zero_nonmax_kernel(int n, float *input, float *output)
{

int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (id >= n) return;

if (input[id] != output[id]) output[id] = 0;
}