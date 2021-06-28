#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void Add_V_V_InPlace(const float* a, int aOffset, float* b, int bOffset, const int n)
{
int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int i = blockId * blockDim.x + threadIdx.x;

if (i < n)
{
b[i + bOffset] = a[i + aOffset] + b[i + bOffset];
}
}