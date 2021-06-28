#include "includes.h"
__global__ void MatrixMul(int *M, int *N, int *P, int width)
{
int accu = 0;

// Block index
int bx = blockIdx.x;
int by = blockIdx.y;

// Thread index
int tx = threadIdx.x;
int ty = threadIdx.y;

int i = by * blockDim.y + ty;
int j = bx * blockDim.x + tx;

for(int k=0; k<width; k++)
{
accu = accu + M[i*width+k]*N[k*width+j];
}

P[i*width+j] = accu;
}