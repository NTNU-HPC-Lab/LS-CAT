#include "includes.h"
__global__ void matrixMulGPU( int * a, int * b, int * c )
{
int val = 0;

int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

if (row < N && col < N)
{
for ( int k = 0; k < N; ++k )
val += a[row * N + k] * b[k * N + col];
c[row * N + col] = val;
}
}