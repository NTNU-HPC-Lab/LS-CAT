#include "includes.h"
__global__ void Subtract(float *d_Result, float *d_Data1, float *d_Data2, int width, int pitch, int height)
{
const int x = blockIdx.x*SUBTRACT_W + threadIdx.x;
const int y = blockIdx.y*SUBTRACT_H + threadIdx.y;
int p = y*pitch + x;
if (x<width && y<height)
d_Result[p] = d_Data1[p] - d_Data2[p];
__syncthreads();
}