#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void FillWithProbabilityMask_V(float* a, const float probability, int n)
{
int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int i = blockId * blockDim.x + threadIdx.x;

if (i < n)
{
float rand = curand_uniform(&randomStates[i % 256]);

a[i] = rand < probability ? 1 : 0;
}
}