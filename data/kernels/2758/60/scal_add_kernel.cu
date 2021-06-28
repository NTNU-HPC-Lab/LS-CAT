#include "includes.h"
__global__ void scal_add_kernel(int N, float ALPHA, float BETA, float *X, int INCX)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i < N) X[i*INCX] = X[i*INCX] * ALPHA + BETA;
}