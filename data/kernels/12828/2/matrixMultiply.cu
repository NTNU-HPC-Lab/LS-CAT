#include "includes.h"
__global__ void matrixMultiply(float* a, float* b, float* c, int n)
{
//use block dimentions to calculate column and row
int col = blockIdx.x*blockDim.x + threadIdx.x;
int row = blockIdx.y*blockDim.y + threadIdx.y;

for(int i = 0; i<n; i++)
{
c[row*n + col]+= a[row*n + i] + b[i*n + col];
}
}