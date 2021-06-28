#include "includes.h"
__global__ void BackwardSoftmax(float *A, float *dA, int nColsdZ, float *dZ)
{
int row = threadIdx.x;
int col = blockIdx.x;

dZ[row * nColsdZ + col] = dA[row * nColsdZ + col] * A[row * nColsdZ + col] *
(1 - A[row * nColsdZ + col]);
}