#include "includes.h"
__global__ void cuda_divide(float * dst, float *numerator, float *denominator, int width, int height)
{
int row = threadIdx.y + blockIdx.y * blockDim.y;
int col = threadIdx.x + blockIdx.x * blockDim.x;

if(row < height && col < width)
{
int index = row * width + col;
if(denominator[index] > 0.0000001)
{
dst[index] = numerator[index] / denominator[index];
}
else
{
dst[index] = 0;
}
//        printf("dst[%d] = %f\n", index, dst[index]);
}
}