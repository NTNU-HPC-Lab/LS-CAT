#include "includes.h"
__global__ void matrixMul(int *a, int *b, int *c, int ROW, int COLUMNS, int temp)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int sum = 0;
if( col < COLUMNS && row < ROW)
{
for(int i = 0; i < temp; i++)
{
sum += a[row * temp + i] * b[i * COLUMNS + col];
}
c[row * COLUMNS + col] = sum;
}


}