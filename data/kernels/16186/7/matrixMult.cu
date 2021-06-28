#include "includes.h"
__global__ void matrixMult (int *a, int *b, int *c, int width)
{
int i, sum = 0;
int col = threadIdx.x + blockDim.x * blockIdx.x;
int row = threadIdx.y + blockDim.y * blockIdx.y;
if(col < width && row < width)
for (i = 0; i< width; i++)
{
sum += a[row * width + i] * b[i * width + col];
}
c[row * width + col] = sum;
}