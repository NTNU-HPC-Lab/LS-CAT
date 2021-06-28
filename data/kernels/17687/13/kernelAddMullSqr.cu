#include "includes.h"
__global__ void kernelAddMullSqr(const int N, double* S, double* A, double m)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N)
{
S[i] += m * A[i] * A[i];
}
}