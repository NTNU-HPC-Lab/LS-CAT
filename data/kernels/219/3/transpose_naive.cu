#include "includes.h"
__global__ void transpose_naive(float *odata, float* idata, int width, int height)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

if (xIndex < width && yIndex < height)
{
unsigned int index_in  = xIndex + width * yIndex;
unsigned int index_out = yIndex + height * xIndex;
odata[index_out] = idata[index_in];
}
}