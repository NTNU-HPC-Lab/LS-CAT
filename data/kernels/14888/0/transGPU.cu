#include "includes.h"


#define BLOCK_SIZE 16


__device__ float f(float x)
{
return 4.f / (1.f + x * x);
}
__global__ void transGPU(const float *inMatrix, float *outMatrix, const size_t row, const size_t column)
{
size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;

if ((xIndex < column) && (yIndex < row))
{
size_t inIndex = yIndex * column + xIndex;
size_t outIndex = xIndex * row + yIndex;

outMatrix[outIndex] = inMatrix[inIndex];
}
}