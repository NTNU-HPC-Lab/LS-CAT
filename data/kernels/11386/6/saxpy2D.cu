#include "includes.h"
__global__ void saxpy2D(float scalar, float * x, float * y)
{
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

if ( row < NX && col < NY ) // Make sure we don't do more work than we have data!
y[row*NY+col] = scalar * x[row*NY+col] + y[row*NY+col];
}