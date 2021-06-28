#include "includes.h"
__global__ void MultiplyAdd(float *d_Result, float *d_Data, int width, int height)
{
const int x = __mul24(blockIdx.x, 16) + threadIdx.x;
const int y = __mul24(blockIdx.y, 16) + threadIdx.y;
int p = __mul24(y, width) + x;
if (x<width && y<height)
d_Result[p] = d_ConstantA[0]*d_Data[p] + d_ConstantB[0];
__syncthreads();
}