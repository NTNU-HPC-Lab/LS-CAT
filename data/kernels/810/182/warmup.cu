#include "includes.h"
__global__ void warmup(int *out, int N) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < N)
{
out[tid] = 0;
}
}