#include "includes.h"
__global__ void setupRandomStatesKernel(curandState* __restrict__ states, unsigned long long seed, int count)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
{
curand_init(seed, i, 0, &states[i]);
}
}