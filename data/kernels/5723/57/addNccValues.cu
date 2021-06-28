#include "includes.h"
__global__ void addNccValues(const float* prevData, float* result, int patches)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < patches)
{
float norm = prevData[3 * tid + 1] * prevData[3 * tid + 2];
float res = 0;
if (norm > 0)
res = prevData[3 * tid] / sqrtf(norm);
result[tid] += res;
}
}