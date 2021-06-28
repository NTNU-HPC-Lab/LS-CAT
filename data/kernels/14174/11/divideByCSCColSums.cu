#include "includes.h"
__global__ void divideByCSCColSums(const float *values, const int *colPointers, float *pixels, const size_t n)
{
const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx >= n)
return;

float weight = 0.0f;

for (size_t ridx = colPointers[idx]; ridx < colPointers[idx+1]; ++ridx)
{
weight += values[ridx];
}

pixels[idx] /= weight + 1e-6f;
}