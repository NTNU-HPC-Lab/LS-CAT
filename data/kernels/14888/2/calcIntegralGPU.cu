#include "includes.h"
__device__ float f(float x)
{
return 4.f / (1.f + x * x);
}
__global__ void calcIntegralGPU(float *array, const float h, const long stepCount, const int threads, const int blocks)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = idx; i < stepCount; i+= threads * blocks)
{
float x = (i + 0.5f) * h;
array[idx] += f(x);
}
}