#include "includes.h"
__global__ void matrixMul(float *M, float *N, float *P, int width)
{
int col= blockDim.x * blockIdx.x + threadIdx.x;
int row = blockDim.y * blockIdx.y + threadIdx.y;
if (row < width && col < width)
{
float pValue = 0;
for(int k=0; k<width; k++)
pValue += M[row * width + k] * N[k * width + col];
P[row * width + col] = pValue;
}
}