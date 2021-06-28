#include "includes.h"

#define N  64


/*
* This CPU function already works, and will run to create a solution matrix
* against which to verify your work building out the matrixMulGPU kernel.
*/

__global__ void matrixMulGPU( int * a, int * b, int * c )
{
/*
* Build out this kernel.
*/
int val = 0;
int row = threadIdx.x + blockIdx.x * blockDim.x;
int col = threadIdx.y + blockIdx.y * blockDim.y;

if (row < N && col < N)
{
for ( int k = 0; k < N; ++k )
val += a[row * N + k] * b[k * N + col];
c[row * N + col] = val;
}
}