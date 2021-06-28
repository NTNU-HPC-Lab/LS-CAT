#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void Sigmoid_V(const float* a, float* out, const int n)
{
int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int i = blockId * blockDim.x + threadIdx.x;

if (i < n)
{
out[i] = 1.0f / (1.0f + __expf(-a[i]));
}
}