#include "includes.h"
__global__ void MyKernel(int *a, int *b, int *c, int N)
{
int idx = threadIdx.x + blockIdx.x * blockDim.x;

if (idx < N) { c[idx] = a[idx] + b[idx]; }
}