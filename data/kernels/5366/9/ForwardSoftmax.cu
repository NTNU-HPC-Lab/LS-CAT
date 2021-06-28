#include "includes.h"
__global__ void ForwardSoftmax(float *Z, int nColsZ, float *sumExp, float *A)
{
int row = threadIdx.x;
int col = blockIdx.x;

atomicAdd(&sumExp[col], exp(Z[row * nColsZ + col]));

__syncthreads();

A[row * nColsZ + col] = exp(Z[row * nColsZ + col]) / sumExp[col];
}