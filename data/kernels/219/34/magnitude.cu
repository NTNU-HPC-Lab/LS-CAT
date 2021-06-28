#include "includes.h"
__global__ void magnitude(float *vec, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if (xIndex < n) { vec[xIndex] = abs(vec[xIndex]); }
}