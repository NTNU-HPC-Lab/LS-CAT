#include "includes.h"
__global__ void gen_matvec(float *A, float *x, float *y, const int m, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
if ( xIndex < m ){
float c = 0.0f;
for(int i=0; i<n; i++)
c = c + x[i] * A[xIndex + m * i];
y[xIndex] = c;
}
}