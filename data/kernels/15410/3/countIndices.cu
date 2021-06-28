#include "includes.h"
__global__ void countIndices(int *indices, unsigned int *histo, int size)
{
int id = threadIdx.x + blockIdx.x * blockDim.x;

int min = blockIdx.x * blockDim.x;
int max = (blockIdx.x + 1) * blockDim.x;

extern __shared__ unsigned int tmp[];

tmp[threadIdx.x] = 0;

__syncthreads();

for (int i = threadIdx.x; i < size; i+=blockDim.x)
{
if (min <= indices[i] && indices[i] < max)
{
atomicAdd(&tmp[indices[i] - min], 1);
}
}

__syncthreads();

histo[id] = tmp[threadIdx.x];

return;
}