#include "includes.h"
__global__ void kernelGetPhi2(const int N, double *T, double *q)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N)
{
T[i] = q[i] * q[i];
}
}