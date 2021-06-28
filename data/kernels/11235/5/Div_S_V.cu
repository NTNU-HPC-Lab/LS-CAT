#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void Div_S_V(const float a, const float* b, float* out, const int n)
{
int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int i = blockId * blockDim.x + threadIdx.x;

if (i < n)
{
out[i] = a / b[i];
}
}