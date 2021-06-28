#include "includes.h"
__global__ void backward_zero_nonmax_kernel(int n, int *indexes, float *prev_delta)
{

int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (id >= n) return;

if (indexes[id] != id) prev_delta[id] = 0;
}