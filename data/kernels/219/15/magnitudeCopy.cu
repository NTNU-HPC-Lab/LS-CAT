#include "includes.h"
__global__ void magnitudeCopy(float *mag_vec, float *vec, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;

if (xIndex < n) { mag_vec[xIndex] = abs(vec[xIndex]); }
}