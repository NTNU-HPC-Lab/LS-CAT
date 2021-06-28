#include "includes.h"
__global__ void cuda_multiply(float *dst, float *src1, float *src2, int width, int height)
{
int row = threadIdx.y + blockIdx.y * blockDim.y;
int col = threadIdx.x + blockIdx.x * blockDim.x;

if(row < height && col < width)
{
int index = row * width + col;
dst[index] = src1[index] * src2[index];
#ifdef debug
printf("multiply dst[%d] = %f\n", index, dst[index]);
#endif
}
}