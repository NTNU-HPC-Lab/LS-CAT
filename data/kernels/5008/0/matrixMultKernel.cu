#include "includes.h"

#define N 1200
#define THREADS 1024


__global__ void matrixMultKernel (double *a, double *b, double *c, int n)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if((row < n) && (col < n)){
double v = 0;
for(int k = 0; k < n; k++){
v += a[row * n + k] * b[k * n + col];
}
c[row * n + col] = v;
}
}