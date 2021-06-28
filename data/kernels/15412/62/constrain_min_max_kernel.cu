#include "includes.h"
__global__ void constrain_min_max_kernel(int N, float MIN, float MAX, float *X, int INCX)
{
int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
if (i < N) X[i*INCX] = fminf(MAX, fmaxf(MIN, X[i*INCX]));
}