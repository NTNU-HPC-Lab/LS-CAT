#include "includes.h"
__global__ void deviceKernel(int *a, int N)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;
int stride = blockDim.x * gridDim.x;

for (int i = idx; i < N; i += stride)
{
a[i] = 1;
}
}