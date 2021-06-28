#include "includes.h"
__global__ void kernelAddConstant(int *g_a, const int b)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
g_a[idx] += b;
}