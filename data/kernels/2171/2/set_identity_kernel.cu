#include "includes.h"
__global__ void set_identity_kernel( float *a, int m, int n )
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if( col < n && row < m)
{
a[row * n + col] = (row == col) ? 1.0f: 0.0f;
}
}