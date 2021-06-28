#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void Add_V_S(const float* a, const float b, float* out, const int n)
{
int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int i = blockId * blockDim.x + threadIdx.x;

if (i < n)
{
out[i] = a[i] + b;
}
}