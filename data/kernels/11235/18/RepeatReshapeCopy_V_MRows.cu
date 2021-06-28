#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void RepeatReshapeCopy_V_MRows(const float* a, float* b, const int rows, const int cols, const int n)
{
int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int i = blockId * blockDim.x + threadIdx.x;

if (i < cols)
{
float value = a[i];

while (i < n)
{
b[i] = value;

i += cols;
}
}
}