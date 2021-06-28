#include "includes.h"
__global__ void quad(float *a, int n, float *u, float *v)
{
int col  = blockIdx.x * blockDim.x + threadIdx.x;
int row  = blockIdx.y * blockDim.y + threadIdx.y;

if (row < n && col < n && col >= row) {
float sum = u[col]*a[row*n+col]*u[row];
if (col == row)
atomicAdd(v, sum);
else
atomicAdd(v, 2*sum);
}
}