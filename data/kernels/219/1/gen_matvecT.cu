#include "includes.h"
__global__ void gen_matvecT(float *A, float *x, float *y, const int m, const int n)
{
unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
if ( xIndex < n ) {
float c = 0.0f;
for(int i=0; i<m; i++)
c = c + y[i] * A[xIndex * m + i];
x[xIndex] = c;
}
}