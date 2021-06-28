#include "includes.h"
__global__ void matMul(float *a, float *b, float *c, int M, int N, int K)
{
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y * blockDim.y + threadIdx.y;

if(row >= M || col >= N)
return;

float sum = 0.f;

__syncthreads();

for(int k = 0; k < K; k++)
{
sum += a[col * K + k] * b[k * N + row];
}

__syncthreads();

c[col * N + row] = sum;
}